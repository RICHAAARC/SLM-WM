"""验证单次方法运行的总科学内容证据身份。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from PIL import Image
import torch

from experiments.protocol.detection_key_identity import (
    REGISTERED_WATERMARK_KEY_ROLE,
    REGISTERED_WRONG_KEY_ROLE,
    build_detection_key_plan_record,
    resolve_detection_key_material_and_identity,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    _align_image,
    _carrier_only_counterfactual_artifact_binding_ready,
    _carrier_only_counterfactual_identity,
    _scientific_content_binding_artifact_ready,
    semantic_watermark_runtime_config_digest,
    semantic_watermark_runtime_config_payload,
)
from experiments.runners.image_only_dataset_runtime import (
    formal_low_frequency_carrier_protocol_record,
)
from experiments.runtime.repository_environment import file_digest
from experiments.runtime.scientific_content_binding import (
    IMAGE_RGB_UINT8_CONTENT_SCHEMA,
    SCIENTIFIC_CONTENT_BINDING_SCHEMA,
    build_scientific_content_binding_record,
    canonical_rgb_uint8_content_record,
    read_canonical_rgb_uint8_content_record,
    recompute_scientific_content_binding_digest,
)
from main.core.digest import TENSOR_CONTENT_DIGEST_VERSION, build_stable_digest
from main.methods.geometry import (
    ATTENTION_ALIGNMENT_ANCHOR_COUNT,
    ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO,
    ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD,
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    ATTENTION_RELATION_COMPONENT_NAMES,
    DIRECT_QK_RELATION_SOURCE,
    attention_alignment_gate_record,
    attention_relation_component_protocol,
    build_qk_atomic_content_metadata,
    qk_atomic_content_records_digest,
    qk_atomic_evaluation_records_digest,
    qk_operator_metadata_records_digest,
)
from main.methods.carrier import (
    keyed_prg_protocol_record,
    tail_robust_carrier_protocol_record,
)
from main.methods.detection import (
    recompute_image_only_detection_digest_payload,
)
from main.methods.method_definition import (
    semantic_conditioned_latent_method_definition_digest,
)
from main.methods.subspace import (
    JACOBIAN_NULL_SPACE_EVIDENCE_VERSION,
    recompute_jacobian_null_space_result_digest,
)
from main.methods.update_composition import (
    QUANTIZED_COMPOSITION_EVIDENCE_VERSION,
    QUANTIZED_COMPOSITION_ORDER,
    recompute_quantized_composition_evidence_digest,
)
from tests.helpers.formal_detection_record import bind_formal_detection_record


_BRANCHES = ("lf_content", "tail_robust", "attention_geometry")
_CARRIER_BRANCHES = ("lf_content", "tail_robust")
_FIXTURE_REGISTERED_KEY_MATERIAL = "scientific-content-binding-key"
_FIXTURE_REGISTERED_KEY_DIGEST_RANDOM = build_stable_digest(
    {"key_material": _FIXTURE_REGISTERED_KEY_MATERIAL}
)
_ATTENTION_LAYER_NAMES = SemanticWatermarkRuntimeConfig().attention_module_names
_LF_CARRIER_PROTOCOL = formal_low_frequency_carrier_protocol_record()
_ATTENTION_ALIGNMENT_GATE = attention_alignment_gate_record(
    ATTENTION_ALIGNMENT_ANCHOR_COUNT,
    ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD,
    ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO,
)
_IDENTITY_AFFINE_TRANSFORM = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
_RISK_SOURCE_FIELDS = (
    "current_decoded_rgb_content_sha256",
    "previous_step_decoded_rgb_content_sha256",
    "clip_patch_tokens_content_sha256",
    "clip_cls_token_content_sha256",
    "semantic_risk_signal_content_sha256",
    "texture_risk_signal_content_sha256",
    "local_contrast_risk_signal_content_sha256",
    "adjacent_step_stability_signal_content_sha256",
    "attention_stability_signal_content_sha256",
)
_RISK_BASE_FIELDS = (
    "risk_values_content_sha256",
    "budget_values_content_sha256",
    "eligible_mask_content_sha256",
)
_RISK_BOUNDED_FIELDS = (
    "effective_budget_values_content_sha256",
    "branch_unit_direction_content_sha256",
    "branch_budget_envelope_content_sha256",
    "branch_written_update_content_sha256",
    "branch_post_risk_direction_content_sha256",
    "branch_post_risk_reference_direction_content_sha256",
    "branch_post_risk_response_content_sha256",
    "branch_post_risk_reference_response_content_sha256",
)
_NULL_SPACE_FIELDS = (
    "candidate_matrix_content_sha256",
    "risk_budget_content_sha256",
    "routed_candidate_response_matrix_content_sha256",
    "projected_direction_matrix_content_sha256",
    "projected_direction_response_matrix_content_sha256",
    "latent_basis_content_sha256",
    "basis_response_matrix_content_sha256",
    "basis_reference_response_matrix_content_sha256",
)


def _sha256(index: int) -> str:
    """生成确定且满足格式约束的测试摘要。"""

    return f"{index:064x}"


_PUBLIC_NOISE_CONTENT_SHA256 = _sha256(8500)
_PUBLIC_NOISE_PRG_PAYLOAD = {
    "keyed_prg_version": "fixture_prg_v1",
    "domain": "public_detection",
    "shape": [1, 16, 4, 4],
}
_PUBLIC_NOISE_PRG_DIGEST = build_stable_digest(
    _PUBLIC_NOISE_PRG_PAYLOAD
)


def _digest_supplier(offset: int) -> Callable[[], str]:
    """返回单调递增的摘要生成器, 避免测试叶子发生意外重名。"""

    cursor = offset

    def take() -> str:
        nonlocal cursor
        cursor += 1
        return _sha256(cursor)

    return take


def _template_identity_record(
    *,
    branch_name: str,
    canonical_template_content_sha256: str,
    embedded_direction_content_sha256: str,
    null_space_digest: str,
    carrier_protocol_digest: str,
    config: SemanticWatermarkRuntimeConfig,
) -> dict[str, Any]:
    """构造不持久化嵌套摘要正文的模板引用."""

    payload = {
        "branch_name": branch_name,
        "template_shape": [1, 16, 4, 4],
        "projection_energy_retention": 0.2,
        "minimum_projection_energy_retention": (
            config.minimum_projection_energy_retention
        ),
        "null_space_digest": null_space_digest,
        "canonical_template_content_sha256": (
            canonical_template_content_sha256
        ),
        "embedded_direction_content_sha256": (
            embedded_direction_content_sha256
        ),
        "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
        "keyed_prg_version": config.keyed_prg_version,
        "keyed_prg_protocol_digest": keyed_prg_protocol_record(
            config.keyed_prg_version
        )["keyed_prg_protocol_digest"],
        "carrier_protocol_digest": carrier_protocol_digest,
    }
    return {
        "branch_name": branch_name,
        "projection_energy_retention": 0.2,
        "carrier_protocol_digest": carrier_protocol_digest,
        "template_shape": [1, 16, 4, 4],
        "canonical_template_content_sha256": (
            canonical_template_content_sha256
        ),
        "template_digest": build_stable_digest(payload),
    }


def _null_space_record(
    branch_name: str,
    take_digest: Callable[[], str],
) -> dict[str, Any]:
    """构造可由独立公式重算 solver 摘要的最小 Null Space 记录。"""

    record: dict[str, Any] = {
        "null_space_evidence_version": JACOBIAN_NULL_SPACE_EVIDENCE_VERSION,
        "branch_name": branch_name,
        "candidate_shape": [16, 2],
        "response_shape": [8, 1],
        "null_rank": 1,
        "evaluated_direction_indices": [0],
        "column_response_norms": [1e-6],
        "column_reference_response_norms": [1.0],
        "column_relative_response_residuals": [1e-6],
        "projection_energy_retentions": [0.5],
        "cg_iteration_counts": [1],
        "cg_relative_residuals": [1e-7],
        "response_residual": 1e-6,
        "relative_response_residual": 1e-6,
        "orthogonality_error": 1e-7,
        "metadata": {"qr_condition_number": 1.0},
        **{field_name: take_digest() for field_name in _NULL_SPACE_FIELDS},
    }
    record["solver_digest"] = (
        recompute_jacobian_null_space_result_digest(record)
    )
    return record


def _evaluation_qk_records(
    roles: tuple[str, ...],
    take_digest: Callable[[], str],
    *,
    public_noise_content: str = "",
    public_noise_prg: str = "",
) -> list[dict[str, Any]]:
    """构造具备冻结角色顺序和有序联合摘要输入的 Q/K 评价记录。"""

    records = []
    for index, role in enumerate(roles):
        values = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4) * (
            0.1 + 0.01 * index
        )
        probabilities = torch.softmax(values, dim=-1)
        atoms = [
            build_qk_atomic_content_metadata(
                layer_name,
                values,
                values,
                values,
                probabilities,
                (0, 1, 2, 3),
            )
            for layer_name in _ATTENTION_LAYER_NAMES
        ]
        record = {
            "qk_evaluation_role": role,
            "evaluation_latent_content_sha256": take_digest(),
            "evaluation_score": 0.1 + 0.01 * index,
            "qk_atomic_content_digest": qk_atomic_content_records_digest(
                atoms
            ),
            "qk_atomic_content_records": atoms,
            "qk_atomic_content_ready": True,
        }
        if public_noise_content:
            record.update(
                {
                    "public_detection_noise_content_sha256": (
                        public_noise_content
                    ),
                    "public_detection_noise_prg_identity_digest": (
                        public_noise_prg
                    ),
                    "public_detection_noise_evaluation_index": index,
                }
            )
        records.append(record)
    return records


def _qk_operator_records() -> list[dict[str, Any]]:
    """构造覆盖冻结层名和二维 token 约定的 Q/K 算子元数据。"""

    return [
        {
            "record_layer_name": layer_name,
            "module_layer_name": layer_name,
            "module_class_name": "FixtureAttention",
            "head_count": 1,
            "head_width": 4,
            "attention_scale": 0.5,
            "attention_scale_source": "inverse_sqrt_head_width",
            "q_normalization_applied": False,
            "k_normalization_applied": False,
            "q_normalization_class": None,
            "k_normalization_class": None,
            "source_token_count": 4,
            "source_grid_side": 2,
            "sampled_token_count": 4,
            "sampled_grid_side": 2,
            "sampled_token_indices": [0, 1, 2, 3],
            "coordinate_convention": ATTENTION_COORDINATE_CONVENTION,
            "grid_align_corners": ATTENTION_GRID_ALIGN_CORNERS,
            "centered_logit_aggregation": (
                "mean_of_per_head_row_centered_sampled_qk_logits"
            ),
            "relation_probability_aggregation": (
                "mean_of_per_head_sampled_image_token_probabilities"
            ),
            "mean_probability_is_softmax_of_mean_logits": False,
        }
        for layer_name in _ATTENTION_LAYER_NAMES
    ]


def _update_record(
    branches: tuple[str, ...],
    *,
    step_index: int,
    digest_offset: int,
    config: SemanticWatermarkRuntimeConfig | None = None,
    latent_before_override: str | None = None,
    lf_template_content_sha256_override: str | None = None,
    tail_template_content_sha256_override: str | None = None,
) -> dict[str, Any]:
    """构造能通过风险、Null Space、量化写回和 Q/K 公式复验的记录。"""

    resolved_config = config or SemanticWatermarkRuntimeConfig()
    take_digest = _digest_supplier(digest_offset)
    risk_sources = {
        field_name: take_digest() for field_name in _RISK_SOURCE_FIELDS
    }
    branch_risk_records: dict[str, dict[str, Any]] = {}
    branch_updates: dict[str, str] = {}
    branch_envelopes: dict[str, str] = {}
    for branch_name in _BRANCHES:
        branch_record = {
            "branch_name": branch_name,
            "risk_field_digest": take_digest(),
            **{
                field_name: take_digest()
                for field_name in _RISK_BASE_FIELDS
            },
        }
        if branch_name in branches:
            branch_record.update(
                {
                    field_name: take_digest()
                    for field_name in _RISK_BOUNDED_FIELDS
                }
            )
            branch_updates[branch_name] = branch_record[
                "branch_written_update_content_sha256"
            ]
            branch_envelopes[branch_name] = branch_record[
                "branch_budget_envelope_content_sha256"
            ]
        branch_risk_records[branch_name] = branch_record
    risk_content_records = {
        branch_name: {
            field_name: branch_risk_records[branch_name][field_name]
            for field_name in (
                _RISK_BASE_FIELDS
                + (_RISK_BOUNDED_FIELDS if branch_name in branches else ())
            )
        }
        for branch_name in _BRANCHES
    }
    latent_before = latent_before_override or take_digest()
    latent_after = take_digest()
    combined_update = take_digest()
    combined_envelope = take_digest()
    written_update = take_digest()
    update_content_by_branch = {
        "lf_content": branch_updates.get("lf_content", take_digest()),
        "tail_robust": branch_updates.get(
            "tail_robust",
            take_digest(),
        ),
        "attention_geometry": branch_updates.get(
            "attention_geometry",
            take_digest(),
        ),
    }
    attention_enabled = "attention_geometry" in branches
    record: dict[str, Any] = {
        "step_index": step_index,
        "watermark_key_material_digest_random": (
            _FIXTURE_REGISTERED_KEY_DIGEST_RANDOM
        ),
        "scheduler_step_timestep": float(step_index),
        "post_step_schedule_index": step_index + 1,
        "timestep": float(step_index) + 0.5,
        "adjacent_step_reference_index": step_index - 1,
        "adjacent_step_reference_latent_content_sha256": take_digest(),
        "adjacent_step_stability_status": (
            "measured_from_immediately_previous_scheduler_step"
        ),
        "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
        "active_carrier_branches": list(branches),
        "metadata": {
            "injection_execution_role": (
                "full_method"
                if attention_enabled
                else "carrier_only_counterfactual"
            ),
            "attention_geometry_enabled": attention_enabled,
            "attention_source": (
                "real_qk_projection"
                if attention_enabled
                else "disabled_attention_geometry"
            ),
        },
        **risk_sources,
        "branch_risk_records": branch_risk_records,
        "branch_risk_bundle_digest": build_stable_digest(
            {
                branch_name: branch_risk_records[branch_name][
                    "risk_field_digest"
                ]
                for branch_name in _BRANCHES
            }
        ),
        "branch_risk_content_digest": build_stable_digest(
            {
                "risk_signal_content_records": risk_sources,
                "branch_risk_content_records": risk_content_records,
            }
        ),
        "null_space_records": {
            branch_name: _null_space_record(branch_name, take_digest)
            for branch_name in branches
        },
        "latent_content_sha256_before": latent_before,
        "latent_content_sha256_after": latent_after,
        "combined_update_content_sha256": combined_update,
        "lf_update_content_sha256": update_content_by_branch[
            "lf_content"
        ],
        "tail_robust_update_content_sha256": update_content_by_branch[
            "tail_robust"
        ],
        "attention_geometry_update_content_sha256": (
            update_content_by_branch["attention_geometry"]
        ),
        "branch_updates_content_digest": build_stable_digest(
            update_content_by_branch
        ),
        "combined_budget_envelope_content_sha256": combined_envelope,
        "quantized_write_update_content_sha256": written_update,
        "quantized_composition_evidence_version": (
            QUANTIZED_COMPOSITION_EVIDENCE_VERSION
        ),
        "quantized_write_original_latent_content_sha256": latent_before,
        "quantized_write_candidate_latent_content_sha256": latent_after,
        "quantized_write_update_dtype": "torch.float16",
        "quantized_write_update_shape": [1, 16, 4, 4],
        "quantized_write_composition_order": list(
            QUANTIZED_COMPOSITION_ORDER
        ),
        "quantized_write_active_branch_order": list(branches),
        "quantized_write_branch_content_identities": {
            branch_name: {
                "branch_written_update_content_sha256": branch_updates[
                    branch_name
                ],
                "branch_budget_envelope_content_sha256": branch_envelopes[
                    branch_name
                ],
            }
            for branch_name in branches
        },
        "quantized_write_common_scale": 1.0,
        "quantized_write_backtracking_factor": 0.5,
        "quantized_write_backtracking_step_count": 0,
        "quantized_write_maximum_envelope_ratio": 0.5,
        "quantized_write_budget_envelope_ready": True,
        "quantized_write_reference_feature_content_sha256": take_digest(),
        "quantized_write_jacobian_response_content_sha256": take_digest(),
        "quantized_write_update_norm": 0.1,
        "quantized_write_jacobian_gate_applicable": True,
        "quantized_write_jacobian_gate_ready": True,
        "quantized_write_relative_jacobian_response": 1e-6,
        "quantized_write_jacobian_status": (
            "measured_from_actual_quantized_latent_delta"
        ),
        "keyed_prg_version": resolved_config.keyed_prg_version,
        "keyed_prg_protocol_digest": keyed_prg_protocol_record(
            resolved_config.keyed_prg_version
        )["keyed_prg_protocol_digest"],
        "attention_module_names": list(
            resolved_config.attention_module_names
        ),
        "attention_coordinate_convention": ATTENTION_COORDINATE_CONVENTION,
        "attention_grid_align_corners": ATTENTION_GRID_ALIGN_CORNERS,
    }
    if "attention_geometry" in branches:
        qk_records = _evaluation_qk_records(
            (
                "latent_before",
                "optimization_content_base_latent",
                "accepted_attention_candidate",
                "actual_written_content_base_latent",
                "actual_written_combined_latent",
            ),
            take_digest,
        )
        operator_records = _qk_operator_records()
        record.update(
            {
                "attention_relation_component_names": list(
                    ATTENTION_RELATION_COMPONENT_NAMES
                ),
                "attention_relation_active_component_names": list(
                    attention_relation_component_protocol(
                        resolved_config.attention_relation_component_weights
                    )["attention_relation_active_component_names"]
                ),
                "attention_relation_component_weights": list(
                    resolved_config.attention_relation_component_weights
                ),
                "attention_relation_component_protocol_digest": (
                    attention_relation_component_protocol(
                        resolved_config.attention_relation_component_weights
                    )["attention_relation_component_protocol_digest"]
                ),
                "attention_qk_atomic_content_records": qk_records,
                "attention_qk_atomic_content_digest": (
                    qk_atomic_evaluation_records_digest(
                        qk_records,
                        "attention_qk_atomic_content_records",
                    )
                ),
                "attention_relation_qk_operator_metadata_records": (
                    operator_records
                ),
                "attention_relation_qk_operator_metadata_digest": (
                    qk_operator_metadata_records_digest(operator_records)
                ),
                "attention_relation_qk_operator_metadata_ready": True,
                "attention_qk_atomic_content_ready": True,
            }
        )
    else:
        record.update(
            {
                **{
                    field_name: None
                    for field_name in (
                        "attention_score_before",
                        "attention_content_base_score",
                        "attention_score_after",
                        "attention_actual_written_content_base_score",
                        "attention_final_combined_score",
                        "attention_score_gain",
                        "attention_applied_update_strength",
                        "attention_backtracking_step_count",
                    )
                },
                **{
                    field_name: ""
                    for field_name in (
                        "attention_update_digest",
                        "attention_update_content_sha256",
                        "attention_update_unit_direction_content_sha256",
                        "stable_token_selection_digest",
                        "stable_pair_weight_identity_digest",
                        "stable_pair_weight_realization_digest",
                        "attention_relation_source",
                        "attention_relation_probability_scope",
                        "attention_relation_component_identity_digest",
                        "attention_relation_keyed_projection_digest",
                        "attention_relation_qk_operator_metadata_digest",
                        "attention_relation_component_protocol_digest",
                        "attention_qk_atomic_content_digest",
                    )
                },
                **{
                    field_name: []
                    for field_name in (
                        "stable_token_indices",
                        "attention_relation_component_names",
                        "attention_relation_active_component_names",
                        "attention_relation_component_weights",
                        "attention_relation_qk_operator_metadata_records",
                        "attention_qk_atomic_content_records",
                    )
                },
                "attention_relation_direct_qk_source_ready": False,
                "attention_relation_qk_operator_metadata_ready": False,
                "attention_qk_atomic_content_ready": False,
            }
        )
    lf_template_content_sha256 = (
        lf_template_content_sha256_override or take_digest()
    )
    tail_template_content_sha256 = (
        tail_template_content_sha256_override or take_digest()
    )
    tail_protocol = tail_robust_carrier_protocol_record(
        resolved_config.tail_fraction,
        prg_version=resolved_config.keyed_prg_version,
    )
    lf_identity = _template_identity_record(
        branch_name="lf_content",
        canonical_template_content_sha256=lf_template_content_sha256,
        embedded_direction_content_sha256=take_digest(),
        null_space_digest=record["null_space_records"]["lf_content"][
            "solver_digest"
        ],
        carrier_protocol_digest=_LF_CARRIER_PROTOCOL[
            "lf_carrier_protocol_digest"
        ],
        config=resolved_config,
    )
    tail_identity = _template_identity_record(
        branch_name="tail_robust",
        canonical_template_content_sha256=tail_template_content_sha256,
        embedded_direction_content_sha256=take_digest(),
        null_space_digest=record["null_space_records"]["tail_robust"][
            "solver_digest"
        ],
        carrier_protocol_digest=tail_protocol[
            "tail_carrier_protocol_digest"
        ],
        config=resolved_config,
    )
    record.update(
        {
            "lf_carrier_protocol_digest": _LF_CARRIER_PROTOCOL[
                "lf_carrier_protocol_digest"
            ],
            "lf_template_content_sha256": lf_template_content_sha256,
            "lf_template_digest": lf_identity["template_digest"],
            "lf_template_shape": lf_identity["template_shape"],
            "lf_projection_energy_retention": 0.2,
            "tail_carrier_protocol_digest": tail_protocol[
                "tail_carrier_protocol_digest"
            ],
            "tail_fraction": resolved_config.tail_fraction,
            "tail_template_content_sha256": (
                tail_template_content_sha256
            ),
            "tail_template_digest": tail_identity["template_digest"],
            "tail_template_shape": tail_identity["template_shape"],
            "tail_template_element_count": 256,
            "tail_selected_element_count": 52,
            "tail_threshold": 1.0,
            "tail_retained_fraction": 52 / 256,
            "tail_projection_energy_retention": 0.2,
        }
    )
    record["quantized_composition_evidence_digest"] = (
        recompute_quantized_composition_evidence_digest(record)
    )
    return record


def _detection_record(
    *,
    sample_role: str,
    detection_key_role: str,
    source_path: str,
    source_file_sha256: str,
    source_rgb_sha256: str,
    evaluated_path: str,
    evaluated_file_sha256: str,
    evaluated_rgb_sha256: str,
    lf_template_content_sha256: str,
    tail_template_content_sha256: str,
    digest_offset: int,
    include_aligned_evaluation: bool = False,
    aligned_rgb_sha256: str | None = None,
    evaluation_index_offset: int = 3,
) -> dict[str, Any]:
    """构造逐次绑定图像、公开噪声和检测 Q/K 的盲检记录。"""

    resolved_config = SemanticWatermarkRuntimeConfig()
    _detection_key_material, detection_key_identity = (
        resolve_detection_key_material_and_identity(
            _FIXTURE_REGISTERED_KEY_MATERIAL,
            detection_key_role,
        )
    )
    take_digest = _digest_supplier(digest_offset)
    public_noise_content = _PUBLIC_NOISE_CONTENT_SHA256
    public_noise_prg_payload = dict(_PUBLIC_NOISE_PRG_PAYLOAD)
    public_noise_prg = _PUBLIC_NOISE_PRG_DIGEST
    qk_roles = (
        ("raw_detection_image", "aligned_detection_image")
        if include_aligned_evaluation
        else ("raw_detection_image",)
    )
    if include_aligned_evaluation and not aligned_rgb_sha256:
        raise ValueError("aligned 评价测试必须提供实际重采样像素摘要")
    qk_records = _evaluation_qk_records(
        qk_roles,
        take_digest,
        public_noise_content=public_noise_content,
        public_noise_prg=public_noise_prg,
    )
    for index, qk_record in enumerate(qk_records):
        qk_record["public_detection_noise_evaluation_index"] = (
            evaluation_index_offset + index
        )
    evidence_records = [
        {
            "public_detection_noise_evaluation_index": (
                evaluation_index_offset + index
            ),
            "public_detection_noise_content_sha256": public_noise_content,
            "public_detection_noise_prg_identity_digest": public_noise_prg,
            "public_detection_noise_prg_identity": {
                **public_noise_prg_payload,
                "public_detection_noise_prg_identity_digest": (
                    public_noise_prg
                ),
            },
            "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
            "public_detection_noise_shape": [1, 16, 4, 4],
            "public_detection_noise_dtype": "torch.float16",
        }
        for index in range(len(qk_records))
    ]
    image_bindings = [
        {
            "qk_evaluation_role": qk_record["qk_evaluation_role"],
            "evaluation_image_rgb_uint8_content_sha256": (
                evaluated_rgb_sha256
                if qk_record["qk_evaluation_role"]
                == "raw_detection_image"
                else aligned_rgb_sha256
            ),
            "qk_atomic_content_digest": qk_record[
                "qk_atomic_content_digest"
            ],
            "public_detection_noise_evaluation_index": (
                evaluation_index_offset + index
            ),
        }
        for index, qk_record in enumerate(qk_records)
    ]
    operator_records = _qk_operator_records()
    lf_score = 0.2
    tail_score = 0.1
    content_score = 0.7 * lf_score + 0.3 * tail_score
    aligned_lf_score = 0.3 if include_aligned_evaluation else None
    aligned_tail_score = 0.2 if include_aligned_evaluation else None
    aligned_content_score = (
        0.7 * aligned_lf_score + 0.3 * aligned_tail_score
        if aligned_lf_score is not None and aligned_tail_score is not None
        else None
    )
    record = {
        "sample_role": sample_role,
        **detection_key_identity,
        "watermark_key_material_digest_random": (
            _FIXTURE_REGISTERED_KEY_DIGEST_RANDOM
        ),
        "attack_id": "none",
        "source_image_path": source_path,
        "source_image_digest": source_file_sha256,
        "source_image_rgb_uint8_content_sha256": source_rgb_sha256,
        "source_image_width": 8,
        "source_image_height": 8,
        "evaluated_image_path": evaluated_path,
        "evaluated_image_digest": evaluated_file_sha256,
        "evaluated_image_rgb_uint8_content_sha256": evaluated_rgb_sha256,
        "evaluated_image_width": 8,
        "evaluated_image_height": 8,
        "attacked_image_digest": "",
        "lf_score": lf_score,
        "tail_robust_score": tail_score,
        "content_score": content_score,
        "lf_weight": 0.7,
        "tail_robust_weight": 0.3,
        "tail_fraction": 0.2,
        "lf_carrier_protocol_digest": _LF_CARRIER_PROTOCOL[
            "lf_carrier_protocol_digest"
        ],
        "lf_template_content_sha256": lf_template_content_sha256,
        "tail_template_content_sha256": tail_template_content_sha256,
        "tail_template_shape": [1, 16, 4, 4],
        "raw_content_margin": content_score,
        "aligned_lf_score": aligned_lf_score,
        "aligned_tail_robust_score": aligned_tail_score,
        "aligned_content_score": aligned_content_score,
        "aligned_content_margin": aligned_content_score,
        "positive_by_content": True,
        "attention_geometry_score": 0.4,
        "raw_attention_geometry_score": 0.4,
        "attention_sync_score": 0.4,
        "registration_confidence": 0.8,
        "geometry_reliable": bool(include_aligned_evaluation),
        "content_failure_reason": "content_positive",
        "rescue_eligible": False,
        "rescue_applied": False,
        "evidence_positive": True,
        "alignment": (
            {
                "affine_transform": [
                    list(row) for row in _IDENTITY_AFFINE_TRANSFORM
                ],
                "alignment_digest": take_digest(),
            }
            if include_aligned_evaluation
            else None
        ),
        "metadata": {
            "content_threshold": 0.0,
            "attention_alignment_gate": dict(_ATTENTION_ALIGNMENT_GATE),
            **_ATTENTION_ALIGNMENT_GATE,
            "stable_token_selection_digest": take_digest(),
            "stable_pair_weight_identity_digest": take_digest(),
            "observed_pair_weight_realization_digest": take_digest(),
            "aligned_pair_weight_realization_digest": (
                take_digest() if include_aligned_evaluation else ""
            ),
            "stable_pair_weight_identity_ready": True,
            "attention_relation_source": DIRECT_QK_RELATION_SOURCE,
            "attention_relation_component_identity_digest": take_digest(),
            "attention_relation_keyed_projection_digest": take_digest(),
            "public_detection_noise_content_sha256": public_noise_content,
            "public_detection_noise_prg_identity_digest": public_noise_prg,
            "public_detection_noise_evidence_records": evidence_records,
            "public_detection_noise_evidence_digest": build_stable_digest(
                {
                    "public_detection_noise_evidence_records": (
                        evidence_records
                    )
                }
            ),
            "detection_qk_atomic_content_records": qk_records,
            "detection_qk_atomic_content_digest": (
                qk_atomic_evaluation_records_digest(
                    qk_records,
                    "detection_qk_atomic_content_records",
                )
            ),
            "detection_qk_image_content_bindings": image_bindings,
            "detection_qk_image_content_binding_digest": (
                build_stable_digest(
                    {
                        "detection_qk_image_content_bindings": (
                            image_bindings
                        )
                    }
                )
            ),
            "attention_relation_qk_operator_metadata_records": (
                operator_records
            ),
            "attention_relation_qk_operator_metadata_digest": (
                qk_operator_metadata_records_digest(operator_records)
            ),
        },
    }
    return bind_formal_detection_record(record)


def _final_observability(
    image_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """把最终三图的规范 RGB 身份绑定到冻结三角色 Q/K 记录。"""

    take_digest = _digest_supplier(9000)
    roles = (
        "final_clean_image",
        "final_carrier_only_image",
        "final_watermarked_image",
    )
    public_noise_content = _PUBLIC_NOISE_CONTENT_SHA256
    public_noise_prg_payload = dict(_PUBLIC_NOISE_PRG_PAYLOAD)
    public_noise_prg = _PUBLIC_NOISE_PRG_DIGEST
    qk_records = _evaluation_qk_records(
        roles,
        take_digest,
        public_noise_content=public_noise_content,
        public_noise_prg=public_noise_prg,
    )
    evidence_records = [
        {
            "public_detection_noise_evaluation_index": index,
            "public_detection_noise_content_sha256": public_noise_content,
            "public_detection_noise_prg_identity_digest": public_noise_prg,
            "public_detection_noise_prg_identity": {
                **public_noise_prg_payload,
                "public_detection_noise_prg_identity_digest": (
                    public_noise_prg
                ),
            },
            "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
            "public_detection_noise_shape": [1, 16, 4, 4],
            "public_detection_noise_dtype": "torch.float16",
        }
        for index in range(3)
    ]
    bindings = [
        {
            "qk_evaluation_role": role,
            "evaluation_image_rgb_uint8_content_sha256": image_records[
                image_role
            ]["image_rgb_uint8_content_sha256"],
            "qk_atomic_content_digest": qk_record[
                "qk_atomic_content_digest"
            ],
            "public_detection_noise_content_sha256": (
                public_noise_content
            ),
            "public_detection_noise_prg_identity_digest": (
                public_noise_prg
            ),
            "public_detection_noise_evaluation_index": index,
        }
        for index, (role, image_role, qk_record) in enumerate(zip(
            roles,
            ("clean_image", "carrier_only_image", "watermarked_image"),
            qk_records,
        ))
    ]
    operator_records = _qk_operator_records()
    return {
        "attention_module_names": list(_ATTENTION_LAYER_NAMES),
        "final_image_qk_atomic_content_records": qk_records,
        "final_image_qk_atomic_content_digest": (
            qk_atomic_evaluation_records_digest(
                qk_records,
                "final_image_qk_atomic_content_records",
            )
        ),
        "final_image_qk_image_content_bindings": bindings,
        "final_image_qk_image_content_binding_digest": build_stable_digest(
            {"final_image_qk_image_content_bindings": bindings}
        ),
        "attention_relation_qk_operator_metadata_records": operator_records,
        "attention_relation_qk_operator_metadata_digest": (
            qk_operator_metadata_records_digest(operator_records)
        ),
        "final_image_public_detection_noise_evidence_records": (
            evidence_records
        ),
        "final_image_public_detection_noise_evidence_digest": (
            build_stable_digest(
                {
                    "final_image_public_detection_noise_evidence_records": (
                        evidence_records
                    )
                }
            )
        ),
        "final_image_public_detection_noise_content_sha256": (
            public_noise_content
        ),
        "final_image_public_detection_noise_prg_identity_digest": (
            public_noise_prg
        ),
        "final_image_public_detection_noise_evidence_ready": True,
    }


def _abstract_image_record(path: str, digest_index: int) -> dict[str, Any]:
    """构造无需图像文件的规范最终图像身份。"""

    return {
        "image_path": path,
        "image_file_sha256": _sha256(digest_index),
        "image_rgb_uint8_content_schema": IMAGE_RGB_UINT8_CONTENT_SCHEMA,
        "image_rgb_uint8_content_sha256": _sha256(digest_index + 1),
        "image_width": 8,
        "image_height": 8,
    }


def _binding_inputs() -> dict[str, Any]:
    """返回可用于纯摘要测试的完整科学内容输入。"""

    images = {
        "clean_image": _abstract_image_record("outputs/test/clean.png", 7001),
        "carrier_only_image": _abstract_image_record(
            "outputs/test/carrier.png", 7003
        ),
        "watermarked_image": _abstract_image_record(
            "outputs/test/watermarked.png", 7005
        ),
    }
    full_update_record = _update_record(
        _BRANCHES,
        step_index=6,
        digest_offset=100,
    )
    lf_template_sha256 = full_update_record[
        "lf_template_content_sha256"
    ]
    tail_template_sha256 = full_update_record[
        "tail_template_content_sha256"
    ]
    carrier_update_record = _update_record(
        _CARRIER_BRANCHES,
        step_index=6,
        digest_offset=1000,
        lf_template_content_sha256_override=lf_template_sha256,
        tail_template_content_sha256_override=tail_template_sha256,
    )
    return {
        "run_id": "scientific-content-binding-test",
        "method_definition_digest": (
            semantic_conditioned_latent_method_definition_digest()
        ),
        "scientific_unit_config_digest": _sha256(2),
        "full_update_records": [full_update_record],
        "carrier_only_update_records": [carrier_update_record],
        "detection_key_plan": build_detection_key_plan_record(
            _FIXTURE_REGISTERED_KEY_MATERIAL
        ),
        "detection_records": [
            _detection_record(
                sample_role="positive_source",
                detection_key_role=REGISTERED_WATERMARK_KEY_ROLE,
                source_path=images["clean_image"]["image_path"],
                source_file_sha256=images["clean_image"][
                    "image_file_sha256"
                ],
                source_rgb_sha256=images["clean_image"][
                    "image_rgb_uint8_content_sha256"
                ],
                evaluated_path=images["watermarked_image"]["image_path"],
                evaluated_file_sha256=images["watermarked_image"][
                    "image_file_sha256"
                ],
                evaluated_rgb_sha256=images["watermarked_image"][
                    "image_rgb_uint8_content_sha256"
                ],
                lf_template_content_sha256=lf_template_sha256,
                tail_template_content_sha256=tail_template_sha256,
                digest_offset=5000,
                include_aligned_evaluation=True,
                aligned_rgb_sha256=images["watermarked_image"][
                    "image_rgb_uint8_content_sha256"
                ],
            ),
            _detection_record(
                sample_role="wrong_key_negative",
                detection_key_role=REGISTERED_WRONG_KEY_ROLE,
                source_path=images["watermarked_image"]["image_path"],
                source_file_sha256=images["watermarked_image"][
                    "image_file_sha256"
                ],
                source_rgb_sha256=images["watermarked_image"][
                    "image_rgb_uint8_content_sha256"
                ],
                evaluated_path=images["watermarked_image"]["image_path"],
                evaluated_file_sha256=images["watermarked_image"][
                    "image_file_sha256"
                ],
                evaluated_rgb_sha256=images["watermarked_image"][
                    "image_rgb_uint8_content_sha256"
                ],
                lf_template_content_sha256=_sha256(60003),
                tail_template_content_sha256=_sha256(60004),
                digest_offset=6000,
                include_aligned_evaluation=True,
                aligned_rgb_sha256=images["watermarked_image"][
                    "image_rgb_uint8_content_sha256"
                ],
                evaluation_index_offset=5,
            ),
        ],
        "final_image_records": images,
        "final_image_attention_observability": _final_observability(images),
        "final_image_preservation": {"semantic_cosine": 0.99},
        "carrier_only_final_image_preservation": {"semantic_cosine": 0.98},
        "carrier_only_counterfactual": {"role": "carrier_only"},
        "attention_geometry_enabled": True,
        "null_space_enabled": True,
        "full_active_branches": _BRANCHES,
        "carrier_only_active_branches": _CARRIER_BRANCHES,
    }


@pytest.mark.quick
def test_scientific_content_binding_digest_is_recomputable() -> None:
    """总摘要必须能够只依赖持久化总记录精确重算。"""

    record = build_scientific_content_binding_record(**_binding_inputs())

    assert record["scientific_content_binding_schema"] == (
        SCIENTIFIC_CONTENT_BINDING_SCHEMA
    )
    assert recompute_scientific_content_binding_digest(record) == record[
        "scientific_content_binding_digest"
    ]


@pytest.mark.quick
@pytest.mark.parametrize(
    "template_field_name",
    (
        "lf_template_content_sha256",
        "tail_template_content_sha256",
    ),
)
def test_scientific_binding_rejects_embedding_detection_template_split(
    template_field_name: str,
) -> None:
    """嵌入端和仅图像检测端不得使用两套固定模板身份."""

    inputs = _binding_inputs()
    detection = deepcopy(inputs["detection_records"][0])
    detection[template_field_name] = "f" * 64
    inputs["detection_records"] = [
        bind_formal_detection_record(detection),
        inputs["detection_records"][1],
    ]

    with pytest.raises(ValueError, match="固定模板身份不一致"):
        build_scientific_content_binding_record(**inputs)


@pytest.mark.quick
@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_key_reuses_registered_template",
        "wrong_key_declares_registered_role",
        "different_detection_key_plan",
        "update_key_digest_split",
    ),
)
def test_scientific_binding_rejects_detection_key_identity_split(
    mutation: str,
) -> None:
    """wrong-key 角色、计划和嵌入注册密钥必须形成同一证据闭环."""

    inputs = _binding_inputs()
    wrong_detection = deepcopy(inputs["detection_records"][1])
    if mutation == "wrong_key_reuses_registered_template":
        registered_detection = inputs["detection_records"][0]
        wrong_detection["lf_template_content_sha256"] = (
            registered_detection["lf_template_content_sha256"]
        )
        wrong_detection["tail_template_content_sha256"] = (
            registered_detection["tail_template_content_sha256"]
        )
        wrong_detection = bind_formal_detection_record(wrong_detection)
    elif mutation == "wrong_key_declares_registered_role":
        _material, identity = resolve_detection_key_material_and_identity(
            _FIXTURE_REGISTERED_KEY_MATERIAL,
            REGISTERED_WATERMARK_KEY_ROLE,
        )
        wrong_detection.update(identity)
    elif mutation == "different_detection_key_plan":
        _material, identity = resolve_detection_key_material_and_identity(
            "different-registered-key",
            REGISTERED_WRONG_KEY_ROLE,
        )
        wrong_detection.update(identity)
    else:
        inputs["full_update_records"][0][
            "watermark_key_material_digest_random"
        ] = "f" * 64
    inputs["detection_records"] = [
        inputs["detection_records"][0],
        wrong_detection,
    ]

    with pytest.raises(ValueError, match="密钥|wrong-key|模板身份"):
        build_scientific_content_binding_record(**inputs)


@pytest.mark.quick
def test_detection_public_noise_uses_shared_global_evaluation_indices() -> None:
    """多条检测必须接受 extractor 的连续全局索引, 不得逐记录归零。"""

    inputs = _binding_inputs()
    first = deepcopy(inputs["detection_records"][0])
    second = deepcopy(inputs["detection_records"][1])

    def set_evaluation_indices(
        record: dict[str, Any],
        indices: tuple[int, ...],
    ) -> None:
        """同步改写 raw/aligned 评价使用的连续全局索引。"""

        metadata = record["metadata"]
        for evidence, index in zip(
            metadata["public_detection_noise_evidence_records"],
            indices,
            strict=True,
        ):
            evidence["public_detection_noise_evaluation_index"] = index
        metadata["public_detection_noise_evidence_digest"] = (
            build_stable_digest(
                {
                    "public_detection_noise_evidence_records": metadata[
                        "public_detection_noise_evidence_records"
                    ]
                }
            )
        )
        for qk_record, index in zip(
            metadata["detection_qk_atomic_content_records"],
            indices,
            strict=True,
        ):
            qk_record["public_detection_noise_evaluation_index"] = index
        metadata["detection_qk_atomic_content_digest"] = (
            qk_atomic_evaluation_records_digest(
                metadata["detection_qk_atomic_content_records"],
                "detection_qk_atomic_content_records",
            )
        )
        for image_binding, index in zip(
            metadata["detection_qk_image_content_bindings"],
            indices,
            strict=True,
        ):
            image_binding["public_detection_noise_evaluation_index"] = index
        metadata["detection_qk_image_content_binding_digest"] = (
            build_stable_digest(
                {
                    "detection_qk_image_content_bindings": metadata[
                        "detection_qk_image_content_bindings"
                    ]
                }
            )
        )
        record["detector_digest"] = build_stable_digest(
            recompute_image_only_detection_digest_payload(record)
        )

    set_evaluation_indices(first, (3, 4))
    set_evaluation_indices(second, (5, 6))
    inputs["detection_records"] = [first, second]

    record = build_scientific_content_binding_record(**inputs)
    assert record["detection_content_identities"][0][
        "public_detection_noise_evaluation_indices"
    ] == [3, 4]
    assert record["detection_content_identities"][1][
        "public_detection_noise_evaluation_indices"
    ] == [5, 6]

    set_evaluation_indices(second, (6, 7))
    with pytest.raises(ValueError, match="连续全局索引"):
        build_scientific_content_binding_record(**inputs)


@pytest.mark.quick
def test_scientific_binding_rejects_mixed_detector_config_identities() -> None:
    """同一科学单元不得把不同阈值配置的检测记录合并为一个结论。"""

    inputs = _binding_inputs()
    drifted = deepcopy(inputs["detection_records"][1])
    drifted["metadata"]["content_threshold"] = 0.01
    drifted["raw_content_margin"] = drifted["content_score"] - 0.01
    drifted["aligned_content_margin"] = (
        drifted["aligned_content_score"] - 0.01
    )
    config_digest = "f" * 64
    drifted["image_only_detector_config_digest"] = config_digest
    drifted["metadata"][
        "image_only_detector_config_digest"
    ] = config_digest
    drifted["detector_digest"] = build_stable_digest(
        recompute_image_only_detection_digest_payload(drifted)
    )
    inputs["detection_records"] = [
        inputs["detection_records"][0],
        drifted,
    ]

    with pytest.raises(ValueError, match="不同盲检配置"):
        build_scientific_content_binding_record(**inputs)


@pytest.mark.quick
@pytest.mark.parametrize(
    "leaf_role",
    (
        "risk",
        "null_space",
        "quantized_write",
        "qk",
        "final_image",
        "final_public_detection_noise",
        "public_detection_noise",
    ),
)
def test_scientific_content_binding_digest_rejects_leaf_tampering(
    leaf_role: str,
) -> None:
    """任一关键科学内容叶子变化都必须破坏既有总摘要。"""

    record = build_scientific_content_binding_record(**_binding_inputs())
    tampered = deepcopy(record)
    full_identity = tampered["full_update_content_identities"][0]
    if leaf_role == "risk":
        full_identity["risk_content_evidence"][
            "risk_signal_content_records"
        ]["semantic_risk_signal_content_sha256"] = _sha256(9501)
    elif leaf_role == "null_space":
        full_identity["null_space_content_records"][0][
            "latent_basis_content_sha256"
        ] = _sha256(9502)
    elif leaf_role == "quantized_write":
        full_identity["update_content_records"][
            "quantized_write_update_content_sha256"
        ] = _sha256(9503)
    elif leaf_role == "qk":
        full_identity["attention_qk_atomic_content_digest"] = _sha256(9504)
    elif leaf_role == "final_image":
        tampered["final_image_content_records"][2][
            "image_rgb_uint8_content_sha256"
        ] = _sha256(9505)
    elif leaf_role == "final_public_detection_noise":
        tampered["final_image_public_detection_noise_identity"][
            "public_detection_noise_content_sha256"
        ] = _sha256(9507)
    else:
        tampered["detection_content_identities"][0][
            "public_detection_noise_content_sha256"
        ] = _sha256(9506)

    assert recompute_scientific_content_binding_digest(tampered) != record[
        "scientific_content_binding_digest"
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """按生产端稳定键序把测试原子写入临时 JSONL。"""

    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _persisted_image_record(path: Path, root: Path) -> dict[str, Any]:
    """从测试 PNG 重建文件字节和规范 RGB 像素双重身份。"""

    return {
        "image_path": path.relative_to(root).as_posix(),
        "image_file_sha256": file_digest(path),
        **read_canonical_rgb_uint8_content_record(path),
    }


def _artifact_fixture(
    tmp_path: Path,
    *,
    runtime_config: SemanticWatermarkRuntimeConfig | None = None,
    relative_output_dir: str = (
        "outputs/scientific_content_binding_test"
    ),
    run_id: str = "artifact-content-binding-test",
) -> tuple[
    SemanticWatermarkRuntimeConfig,
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
]:
    """物化可由消费者从磁盘完整重建的最小方法产物。"""

    config = runtime_config or SemanticWatermarkRuntimeConfig()
    output_dir = tmp_path / relative_output_dir
    output_dir.mkdir(parents=True)
    paths = {
        "full_update": output_dir / "latent_update_records.jsonl",
        "carrier_update": output_dir / "carrier_update_records.jsonl",
        "detection": output_dir / "image_only_detection_records.jsonl",
        "clean_image": output_dir / "clean_image.png",
        "carrier_only_image": output_dir / "carrier_only_image.png",
        "watermarked_image": output_dir / "watermarked_image.png",
    }
    for role_index, role in enumerate(
        ("clean_image", "carrier_only_image", "watermarked_image")
    ):
        Image.new(
            "RGB",
            (8, 8),
            color=(32 * role_index, 64, 128),
        ).save(paths[role])
    images = {
        role: _persisted_image_record(paths[role], tmp_path)
        for role in (
            "clean_image",
            "carrier_only_image",
            "watermarked_image",
        )
    }
    with Image.open(paths["watermarked_image"]) as evaluated_image:
        aligned_rgb_sha256 = canonical_rgb_uint8_content_record(
            _align_image(
                evaluated_image,
                SimpleNamespace(
                    affine_transform=_IDENTITY_AFFINE_TRANSFORM
                ),
            )
        )["image_rgb_uint8_content_sha256"]
    shared_initial_latent_sha256 = _sha256(60000)
    shared_lf_template_sha256 = _sha256(60001)
    shared_tail_template_sha256 = _sha256(60002)
    full_records = [
        _update_record(
            _BRANCHES,
            step_index=step_index,
            digest_offset=1000 * (record_index + 1),
            config=config,
            latent_before_override=(
                shared_initial_latent_sha256
                if record_index == 0
                else None
            ),
            lf_template_content_sha256_override=(
                shared_lf_template_sha256
            ),
            tail_template_content_sha256_override=(
                shared_tail_template_sha256
            ),
        )
        for record_index, step_index in enumerate(
            config.injection_step_indices
        )
    ]
    carrier_records = [
        _update_record(
            _CARRIER_BRANCHES,
            step_index=step_index,
            digest_offset=10000 + 1000 * (record_index + 1),
            config=config,
            latent_before_override=(
                shared_initial_latent_sha256
                if record_index == 0
                else None
            ),
            lf_template_content_sha256_override=(
                shared_lf_template_sha256
            ),
            tail_template_content_sha256_override=(
                shared_tail_template_sha256
            ),
        )
        for record_index, step_index in enumerate(
            config.injection_step_indices
        )
    ]
    detection_records = [
        _detection_record(
            sample_role="positive_source",
            detection_key_role=REGISTERED_WATERMARK_KEY_ROLE,
            source_path=images["clean_image"]["image_path"],
            source_file_sha256=images["clean_image"]["image_file_sha256"],
            source_rgb_sha256=images["clean_image"][
                "image_rgb_uint8_content_sha256"
            ],
            evaluated_path=images["watermarked_image"]["image_path"],
            evaluated_file_sha256=images["watermarked_image"][
                "image_file_sha256"
            ],
            evaluated_rgb_sha256=images["watermarked_image"][
                "image_rgb_uint8_content_sha256"
            ],
            lf_template_content_sha256=shared_lf_template_sha256,
            tail_template_content_sha256=shared_tail_template_sha256,
            digest_offset=50000,
            include_aligned_evaluation=True,
            aligned_rgb_sha256=aligned_rgb_sha256,
        ),
        _detection_record(
            sample_role="wrong_key_negative",
            detection_key_role=REGISTERED_WRONG_KEY_ROLE,
            source_path=images["watermarked_image"]["image_path"],
            source_file_sha256=images["watermarked_image"][
                "image_file_sha256"
            ],
            source_rgb_sha256=images["watermarked_image"][
                "image_rgb_uint8_content_sha256"
            ],
            evaluated_path=images["watermarked_image"]["image_path"],
            evaluated_file_sha256=images["watermarked_image"][
                "image_file_sha256"
            ],
            evaluated_rgb_sha256=images["watermarked_image"][
                "image_rgb_uint8_content_sha256"
            ],
            lf_template_content_sha256=_sha256(60003),
            tail_template_content_sha256=_sha256(60004),
            digest_offset=51000,
            include_aligned_evaluation=True,
            aligned_rgb_sha256=aligned_rgb_sha256,
            evaluation_index_offset=5,
        ),
    ]
    _write_jsonl(paths["full_update"], full_records)
    _write_jsonl(paths["carrier_update"], carrier_records)
    _write_jsonl(paths["detection"], detection_records)
    relative_paths = {
        role: path.relative_to(tmp_path).as_posix()
        for role, path in paths.items()
    }
    counterfactual = _carrier_only_counterfactual_identity(
        config,
        replace(config, attention_geometry_enabled=False),
        full_records,
        carrier_records,
    )
    counterfactual.update(
        {
        "carrier_only_counterfactual_atom_path": relative_paths[
            "carrier_update"
        ],
        "carrier_only_counterfactual_image_path": relative_paths[
            "carrier_only_image"
        ],
            "carrier_only_counterfactual_atom_file_sha256": file_digest(
                paths["carrier_update"]
            ),
            "carrier_only_counterfactual_image_digest": file_digest(
                paths["carrier_only_image"]
            ),
        }
    )
    observability = _final_observability(images)
    preservation = {
        "semantic_cosine": 0.99,
        "final_image_preservation_gate_ready": True,
    }
    carrier_preservation = {
        "semantic_cosine": 0.98,
        "carrier_only_final_image_preservation_gate_ready": True,
        "carrier_only_to_full_final_image_preservation_gate_ready": True,
        "carrier_only_counterfactual_three_way_preservation_gate_ready": (
            True
        ),
    }
    carrier_artifact_identity = {
        field_name: counterfactual[field_name]
        for field_name in (
            "carrier_only_counterfactual_identity_digest",
            "carrier_only_counterfactual_image_path",
            "carrier_only_counterfactual_image_digest",
            "carrier_only_counterfactual_atom_path",
            "carrier_only_counterfactual_atom_file_sha256",
            "carrier_only_counterfactual_atom_content_digest",
        )
    }
    observability.update(carrier_artifact_identity)
    carrier_preservation.update(carrier_artifact_identity)
    detection_key_plan = build_detection_key_plan_record(
        _FIXTURE_REGISTERED_KEY_MATERIAL
    )
    binding_record = build_scientific_content_binding_record(
        run_id=run_id,
        method_definition_digest=(
            semantic_conditioned_latent_method_definition_digest()
        ),
        scientific_unit_config_digest=(
            semantic_watermark_runtime_config_digest(config)
        ),
        full_update_records=full_records,
        carrier_only_update_records=carrier_records,
        detection_records=detection_records,
        detection_key_plan=detection_key_plan,
        final_image_records=images,
        final_image_attention_observability=observability,
        final_image_preservation=preservation,
        carrier_only_final_image_preservation=carrier_preservation,
        carrier_only_counterfactual=counterfactual,
        attention_geometry_enabled=True,
        null_space_enabled=True,
        full_active_branches=_BRANCHES,
        carrier_only_active_branches=_CARRIER_BRANCHES,
    )
    result_payload = {
        "run_id": run_id,
        "update_record_path": relative_paths["full_update"],
        "detection_record_path": relative_paths["detection"],
        "clean_image_path": relative_paths["clean_image"],
        "watermarked_image_path": relative_paths["watermarked_image"],
        "metadata": {
            "scientific_content_binding_schema": (
                SCIENTIFIC_CONTENT_BINDING_SCHEMA
            ),
            "scientific_content_binding_record": binding_record,
            "scientific_content_binding_digest": binding_record[
                "scientific_content_binding_digest"
            ],
            "final_image_attention_observability": observability,
            "final_image_preservation": preservation,
            "carrier_only_final_image_preservation": carrier_preservation,
            "carrier_only_counterfactual": counterfactual,
        },
    }
    manifest = {
        "output_paths": list(relative_paths.values()),
        "metadata": {
            "scientific_content_binding_schema": (
                SCIENTIFIC_CONTENT_BINDING_SCHEMA
            ),
            "scientific_content_binding_digest": binding_record[
                "scientific_content_binding_digest"
            ],
            "detection_key_plan": detection_key_plan,
            **carrier_artifact_identity,
        },
    }
    return config, result_payload, manifest, paths


@pytest.mark.quick
@pytest.mark.parametrize(
    "leaf_role",
    (
        "risk",
        "null_space",
        "quantized_write",
        "qk",
        "final_image",
        "final_public_detection_noise",
        "public_detection_noise",
        "aligned_detection_image",
    ),
)
def test_scientific_content_binding_artifact_validator_rejects_tampering(
    tmp_path: Path,
    leaf_role: str,
) -> None:
    """消费者必须从文件重建身份并拒绝任一关键内容被替换。"""

    config, result_payload, manifest, paths = _artifact_fixture(tmp_path)
    assert _scientific_content_binding_artifact_ready(
        result_payload,
        manifest,
        tmp_path,
        config,
    )

    if leaf_role == "final_image":
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(
            paths["watermarked_image"]
        )
    elif leaf_role == "final_public_detection_noise":
        result_payload["metadata"][
            "final_image_attention_observability"
        ]["final_image_public_detection_noise_content_sha256"] = (
            _sha256(9906)
        )
    elif leaf_role in {
        "public_detection_noise",
        "aligned_detection_image",
    }:
        detection_records = [
            json.loads(line)
            for line in paths["detection"].read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        detection_record = detection_records[0]
        if leaf_role == "public_detection_noise":
            detection_record["metadata"][
                "public_detection_noise_content_sha256"
            ] = _sha256(9901)
        else:
            detection_record["alignment"]["affine_transform"][0][2] = (
                0.25
            )
        _write_jsonl(paths["detection"], detection_records)
    else:
        update_records = [
            json.loads(line)
            for line in paths["full_update"].read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        first = update_records[0]
        if leaf_role == "risk":
            first["semantic_risk_signal_content_sha256"] = _sha256(9902)
        elif leaf_role == "null_space":
            first["null_space_records"]["lf_content"][
                "latent_basis_content_sha256"
            ] = _sha256(9903)
        elif leaf_role == "quantized_write":
            first["quantized_write_update_content_sha256"] = _sha256(9904)
        else:
            first["attention_qk_atomic_content_digest"] = _sha256(9905)
        _write_jsonl(paths["full_update"], update_records)

    assert not _scientific_content_binding_artifact_ready(
        result_payload,
        manifest,
        tmp_path,
        config,
    )


@pytest.mark.quick
def test_carrier_only_artifact_validator_accepts_persisted_config_payload(
    tmp_path: Path,
) -> None:
    """打包器必须以脱敏配置重建 carrier-only 身份并拒绝字段残留。"""

    config, result_payload, manifest, paths = _artifact_fixture(tmp_path)
    config_payload = semantic_watermark_runtime_config_payload(config)
    assert _carrier_only_counterfactual_artifact_binding_ready(
        result_payload,
        manifest,
        tmp_path,
        config_payload,
    )

    carrier_records = [
        json.loads(line)
        for line in paths["carrier_update"].read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    carrier_records[0]["attention_score_before"] = 0.0
    _write_jsonl(paths["carrier_update"], carrier_records)
    assert not _carrier_only_counterfactual_artifact_binding_ready(
        result_payload,
        manifest,
        tmp_path,
        config_payload,
    )
