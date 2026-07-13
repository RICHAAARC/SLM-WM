"""构造并重建单次方法运行的总科学内容证据身份。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from pathlib import Path
import struct
from typing import Any

from experiments.protocol.detection_key_identity import (
    REGISTERED_WATERMARK_KEY_ROLE,
    REGISTERED_WRONG_KEY_ROLE,
    validate_detection_key_identity_record,
    validate_detection_key_plan_record,
)
from main.core.digest import (
    TENSOR_CONTENT_DIGEST_VERSION,
    build_stable_digest,
)
from main.methods.detection import validate_image_only_detection_digest_record
from main.methods.geometry import (
    qk_atomic_evaluation_records_digest,
    qk_atomic_evaluation_records_ready,
    qk_operator_metadata_records_digest,
    qk_operator_metadata_records_ready,
)
from main.methods.subspace import (
    recompute_jacobian_null_space_result_digest,
)
from main.methods.update_composition import (
    recompute_quantized_composition_evidence_digest,
)


SCIENTIFIC_CONTENT_BINDING_SCHEMA = (
    "slm_wm_scientific_content_binding_v5"
)
IMAGE_RGB_UINT8_CONTENT_SCHEMA = "slm_wm_image_rgb_uint8_content_v1"

_BRANCH_NAMES = (
    "lf_content",
    "tail_robust",
    "attention_geometry",
)
_RISK_SOURCE_CONTENT_FIELDS = (
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
_BRANCH_RISK_BASE_CONTENT_FIELDS = (
    "risk_values_content_sha256",
    "budget_values_content_sha256",
    "eligible_mask_content_sha256",
)
_BRANCH_RISK_ENVELOPE_CONTENT_FIELDS = (
    "effective_budget_values_content_sha256",
    "branch_unit_direction_content_sha256",
    "branch_budget_envelope_content_sha256",
    "branch_written_update_content_sha256",
)
_BRANCH_RISK_POST_NULL_CONTENT_FIELDS = (
    "branch_post_risk_direction_content_sha256",
    "branch_post_risk_reference_direction_content_sha256",
    "branch_post_risk_response_content_sha256",
    "branch_post_risk_reference_response_content_sha256",
)
_UPDATE_CONTENT_FIELDS = (
    "latent_content_sha256_before",
    "latent_content_sha256_after",
    "combined_update_content_sha256",
    "lf_update_content_sha256",
    "tail_robust_update_content_sha256",
    "attention_geometry_update_content_sha256",
    "branch_updates_content_digest",
    "branch_risk_bundle_digest",
    "branch_risk_content_digest",
    "combined_budget_envelope_content_sha256",
    "quantized_write_update_content_sha256",
    "quantized_composition_evidence_digest",
)
_QUANTIZED_WRITE_JACOBIAN_CONTENT_FIELDS = (
    "quantized_write_reference_feature_content_sha256",
    "quantized_write_jacobian_response_content_sha256",
)
_NULL_SPACE_CONTENT_FIELDS = (
    "candidate_matrix_content_sha256",
    "risk_budget_content_sha256",
    "routed_candidate_response_matrix_content_sha256",
    "projected_direction_matrix_content_sha256",
    "projected_direction_response_matrix_content_sha256",
    "latent_basis_content_sha256",
    "basis_response_matrix_content_sha256",
    "basis_reference_response_matrix_content_sha256",
)


def _sha256(value: Any, *, field_name: str) -> str:
    """读取规范 SHA-256, 拒绝对象字符串化形成的伪摘要。"""

    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} 必须为规范 SHA-256")
    return value


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    """把记录转为普通字典并集中拒绝缺失对象。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} 必须为对象")
    return dict(value)


def _public_noise_evidence_identity(
    evidence_records: Any,
    supplied_digest: Any,
    *,
    aggregate_field_name: str,
    expected_indices: Sequence[int],
) -> dict[str, Any]:
    """集中复验公开噪声 Tensor、PRG 身份和全局评价索引。"""

    if (
        not isinstance(evidence_records, list)
        or len(evidence_records) != len(expected_indices)
    ):
        raise ValueError("公开噪声记录数量与冻结评价角色不一致")
    evidence_digest = _sha256(
        supplied_digest,
        field_name=f"{aggregate_field_name}_digest",
    )
    if build_stable_digest(
        {aggregate_field_name: evidence_records}
    ) != evidence_digest:
        raise ValueError("公开噪声摘要不能由评价记录重建")
    content_digests: set[str] = set()
    prg_digests: set[str] = set()
    resolved_indices: list[int] = []
    for record in evidence_records:
        evidence_record = _mapping(
            record,
            field_name=aggregate_field_name,
        )
        evaluation_index = evidence_record.get(
            "public_detection_noise_evaluation_index"
        )
        if (
            isinstance(evaluation_index, bool)
            or not isinstance(evaluation_index, int)
            or evaluation_index < 0
        ):
            raise ValueError("公开噪声评价索引必须为非负整数")
        content_digest = _sha256(
            evidence_record.get(
                "public_detection_noise_content_sha256"
            ),
            field_name="public_detection_noise_content_sha256",
        )
        prg_digest = _sha256(
            evidence_record.get(
                "public_detection_noise_prg_identity_digest"
            ),
            field_name="public_detection_noise_prg_identity_digest",
        )
        prg_identity = _mapping(
            evidence_record.get("public_detection_noise_prg_identity"),
            field_name="public_detection_noise_prg_identity",
        )
        nested_prg_digest = prg_identity.pop(
            "public_detection_noise_prg_identity_digest",
            None,
        )
        evidence_shape = evidence_record.get(
            "public_detection_noise_shape"
        )
        prg_shape = prg_identity.get("shape")
        if (
            nested_prg_digest != prg_digest
            or build_stable_digest(prg_identity) != prg_digest
            or evidence_record.get("tensor_content_digest_version")
            != TENSOR_CONTENT_DIGEST_VERSION
            or not isinstance(evidence_shape, (list, tuple))
            or not isinstance(prg_shape, (list, tuple))
            or tuple(evidence_shape) != tuple(prg_shape)
            or not isinstance(
                evidence_record.get("public_detection_noise_dtype"),
                str,
            )
        ):
            raise ValueError("公开噪声 PRG、shape 或 dtype 身份不一致")
        content_digests.add(content_digest)
        prg_digests.add(prg_digest)
        resolved_indices.append(evaluation_index)
    if (
        resolved_indices != list(expected_indices)
        or len(content_digests) != 1
        or len(prg_digests) != 1
    ):
        raise ValueError("公开噪声必须共享内容和 PRG 并使用冻结全局索引")
    return {
        "public_detection_noise_content_sha256": next(
            iter(content_digests)
        ),
        "public_detection_noise_prg_identity_digest": next(
            iter(prg_digests)
        ),
        "public_detection_noise_evidence_digest": evidence_digest,
        "public_detection_noise_evidence_records_digest": (
            build_stable_digest(evidence_records)
        ),
        "public_detection_noise_evaluation_indices": resolved_indices,
    }


def canonical_rgb_uint8_content_record(image: Any) -> dict[str, Any]:
    """构造与 PNG 编码无关的规范 RGB uint8 像素身份。"""

    rgb = image.convert("RGB")
    width, height = (int(value) for value in rgb.size)
    pixels = rgb.tobytes()
    if len(pixels) != width * height * 3:
        raise ValueError("规范 RGB 图像的像素字节数与尺寸不一致")
    digest = hashlib.sha256()
    digest.update(IMAGE_RGB_UINT8_CONTENT_SCHEMA.encode("ascii"))
    digest.update(struct.pack(">QQ", width, height))
    digest.update(pixels)
    return {
        "image_rgb_uint8_content_schema": IMAGE_RGB_UINT8_CONTENT_SCHEMA,
        "image_rgb_uint8_content_sha256": digest.hexdigest(),
        "image_width": width,
        "image_height": height,
    }


def read_canonical_rgb_uint8_content_record(
    path: str | Path,
) -> dict[str, Any]:
    """从持久化图像文件重建规范 RGB uint8 像素身份。"""

    from PIL import Image

    with Image.open(path) as image:
        return canonical_rgb_uint8_content_record(image)


def _branch_risk_content_evidence(
    resolved: Mapping[str, Any],
    branch_risk_records: Mapping[str, Mapping[str, Any]],
    expected_branches: Sequence[str],
    *,
    semantic_routing_enabled: bool,
    null_space_enabled: bool,
) -> dict[str, Any]:
    """按机制开关重建活动分支风险内容并拒绝禁用残留."""

    composition_identities = _mapping(
        resolved.get("quantized_write_branch_content_identities"),
        field_name="quantized_write_branch_content_identities",
    )
    if set(composition_identities) != set(expected_branches):
        raise ValueError("量化写回的活动分支顺序与方法分支不一致")
    if not semantic_routing_enabled:
        if branch_risk_records or any(
            resolved.get(field_name) not in {None, ""}
            for field_name in _RISK_SOURCE_CONTENT_FIELDS
        ):
            raise ValueError("风险路由关闭时不得保留风险信号或分支风险原子")
        expected_bundle = build_stable_digest(
            {
                "semantic_routing_enabled": False,
                "active_carrier_branches": tuple(expected_branches),
            }
        )
        evidence = {
            "semantic_routing_enabled": False,
            "active_carrier_branches": list(expected_branches),
            "risk_signal_content_records": {},
            "branch_risk_content_records": {},
        }
        if (
            resolved.get("branch_risk_bundle_digest") != expected_bundle
            or resolved.get("branch_risk_content_digest")
            != build_stable_digest(evidence)
        ):
            raise ValueError("关闭风险路由后的空风险身份不能重建")
        return evidence

    required_source_fields = _RISK_SOURCE_CONTENT_FIELDS[:-1]
    risk_sources = {
        field_name: _sha256(
            resolved.get(field_name),
            field_name=field_name,
        )
        for field_name in required_source_fields
    }
    if resolved.get(_RISK_SOURCE_CONTENT_FIELDS[-1]) not in {None, ""}:
        risk_sources[_RISK_SOURCE_CONTENT_FIELDS[-1]] = _sha256(
            resolved.get(_RISK_SOURCE_CONTENT_FIELDS[-1]),
            field_name=_RISK_SOURCE_CONTENT_FIELDS[-1],
        )
    if set(branch_risk_records) != set(expected_branches):
        raise ValueError("风险记录必须精确覆盖活动载体分支")
    branch_content: dict[str, dict[str, Any]] = {}
    update_fields = {
        "lf_content": "lf_update_content_sha256",
        "tail_robust": "tail_robust_update_content_sha256",
        "attention_geometry": "attention_geometry_update_content_sha256",
    }
    for branch_name in expected_branches:
        branch_record = _mapping(
            branch_risk_records[branch_name],
            field_name=f"branch_risk_records.{branch_name}",
        )
        if branch_record.get("branch_name") != branch_name:
            raise ValueError("风险记录的分支名称与冻结角色不一致")
        _sha256(
            branch_record.get("risk_field_digest"),
            field_name=f"{branch_name}.risk_field_digest",
        )
        content = {
            field_name: _sha256(
                branch_record.get(field_name),
                field_name=f"{branch_name}.{field_name}",
            )
            for field_name in _BRANCH_RISK_BASE_CONTENT_FIELDS
        }
        bounded = {
            field_name: _sha256(
                branch_record.get(field_name),
                field_name=f"{branch_name}.{field_name}",
            )
            for field_name in _BRANCH_RISK_ENVELOPE_CONTENT_FIELDS
        }
        if null_space_enabled:
            bounded.update(
                {
                    field_name: _sha256(
                        branch_record.get(field_name),
                        field_name=f"{branch_name}.{field_name}",
                    )
                    for field_name in _BRANCH_RISK_POST_NULL_CONTENT_FIELDS
                }
            )
        elif any(
            field_name in branch_record
            for field_name in _BRANCH_RISK_POST_NULL_CONTENT_FIELDS
        ):
            raise ValueError("Null Space 关闭时不得保留投影后风险响应原子")
        composition_identity = _mapping(
            composition_identities[branch_name],
            field_name=f"quantized_write_branch_content_identities.{branch_name}",
        )
        if composition_identity != {
            "branch_written_update_content_sha256": bounded[
                "branch_written_update_content_sha256"
            ],
            "branch_budget_envelope_content_sha256": bounded[
                "branch_budget_envelope_content_sha256"
            ],
        }:
            raise ValueError("风险记录与量化写回的分支身份不一致")
        if bounded["branch_written_update_content_sha256"] != resolved.get(
            update_fields[branch_name]
        ):
            raise ValueError("风险记录与顶层分支 update 身份不一致")
        content.update(bounded)
        branch_content[branch_name] = content
    expected_bundle = build_stable_digest(
        {
            branch_name: branch_risk_records[branch_name][
                "risk_field_digest"
            ]
            for branch_name in expected_branches
        }
    )
    evidence = {
        "semantic_routing_enabled": True,
        "active_carrier_branches": list(expected_branches),
        "risk_signal_content_records": risk_sources,
        "branch_risk_content_records": branch_content,
    }
    if resolved.get("branch_risk_bundle_digest") != expected_bundle:
        raise ValueError("活动风险场联合摘要不能由分支记录重建")
    if resolved.get("branch_risk_content_digest") != build_stable_digest(
        evidence
    ):
        raise ValueError("风险来源和分支内容摘要不能由叶子身份重建")
    return evidence


def _update_content_identity(
    record: Mapping[str, Any],
    *,
    expected_branches: Sequence[str],
    null_space_enabled: bool,
    semantic_routing_enabled: bool,
) -> dict[str, Any]:
    """提取一次注入原子的风险、基底、分支、写回与 Q/K 身份。"""

    resolved = dict(record)
    resolved_branches = list(expected_branches)
    if (
        tuple(resolved_branches)
        != tuple(name for name in _BRANCH_NAMES if name in resolved_branches)
        or len(set(resolved_branches)) != len(resolved_branches)
        or not resolved_branches
    ):
        raise ValueError("活动载体分支必须是冻结顺序的非空子集")
    metadata = _mapping(resolved.get("metadata"), field_name="update.metadata")
    if metadata.get("semantic_routing_enabled") is not semantic_routing_enabled:
        raise ValueError("注入记录的风险路由开关与运行配置不一致")
    watermark_key_material_digest_random = _sha256(
        resolved.get("watermark_key_material_digest_random"),
        field_name="watermark_key_material_digest_random",
    )
    lf_carrier_protocol_digest = _sha256(
        resolved.get("lf_carrier_protocol_digest"),
        field_name="lf_carrier_protocol_digest",
    )
    tail_carrier_protocol_digest = _sha256(
        resolved.get("tail_carrier_protocol_digest"),
        field_name="tail_carrier_protocol_digest",
    )
    lf_template_shape = list(resolved.get("lf_template_shape", ()))
    tail_template_shape = list(resolved.get("tail_template_shape", ()))
    tail_template_element_count = resolved.get("tail_template_element_count")
    tail_selected_element_count = resolved.get("tail_selected_element_count")
    tail_fraction = resolved.get("tail_fraction")
    if "lf_content" in resolved_branches:
        lf_template_content_sha256 = _sha256(
            resolved.get("lf_template_content_sha256"),
            field_name="lf_template_content_sha256",
        )
        if len(lf_template_shape) != 4 or any(
            type(value) is not int or value <= 0
            for value in lf_template_shape
        ):
            raise ValueError("LF 模板 shape 无效")
        lf_template_digest = _sha256(
            resolved.get("lf_template_digest"),
            field_name="lf_template_digest",
        )
        lf_projection_energy_retention = resolved.get(
            "lf_projection_energy_retention"
        )
        if (
            not isinstance(lf_projection_energy_retention, float)
            or not math.isfinite(lf_projection_energy_retention)
        ):
            raise ValueError("LF 投影能量比例无效")
    else:
        lf_template_content_sha256 = ""
        lf_template_digest = ""
        lf_projection_energy_retention = None
        if (
            resolved.get("lf_template_content_sha256") != ""
            or resolved.get("lf_template_digest") != ""
            or lf_template_shape != []
            or resolved.get("lf_projection_energy_retention") is not None
        ):
            raise ValueError("禁用 LF 分支不得保留模板原子")
    if "tail_robust" in resolved_branches:
        tail_template_content_sha256 = _sha256(
            resolved.get("tail_template_content_sha256"),
            field_name="tail_template_content_sha256",
        )
        if (
            len(tail_template_shape) != 4
            or any(
                type(value) is not int or value <= 0
                for value in tail_template_shape
            )
            or type(tail_template_element_count) is not int
            or tail_template_element_count != math.prod(tail_template_shape)
            or type(tail_selected_element_count) is not int
            or type(tail_fraction) is not float
            or tail_selected_element_count
            != math.ceil(tail_template_element_count * tail_fraction)
        ):
            raise ValueError("尾部模板 shape 或选择计数无效")
        tail_template_digest = _sha256(
            resolved.get("tail_template_digest"),
            field_name="tail_template_digest",
        )
        tail_projection_energy_retention = resolved.get(
            "tail_projection_energy_retention"
        )
        if (
            not isinstance(tail_projection_energy_retention, float)
            or not math.isfinite(tail_projection_energy_retention)
        ):
            raise ValueError("尾部投影能量比例无效")
    else:
        tail_template_content_sha256 = ""
        tail_template_digest = ""
        tail_projection_energy_retention = None
        if (
            resolved.get("tail_template_content_sha256") != ""
            or resolved.get("tail_template_digest") != ""
            or tail_template_shape != []
            or tail_template_element_count != 0
            or tail_selected_element_count != 0
            or resolved.get("tail_threshold") != 0.0
            or resolved.get("tail_retained_fraction") != 0.0
            or resolved.get("tail_projection_energy_retention") is not None
        ):
            raise ValueError("禁用尾部分支不得保留模板原子")
    if resolved.get("tensor_content_digest_version") != (
        TENSOR_CONTENT_DIGEST_VERSION
    ):
        raise ValueError("注入原子使用了错误的 Tensor 内容摘要版本")
    if resolved.get("active_carrier_branches") != resolved_branches:
        raise ValueError("注入原子的活动分支与方法角色不一致")
    branch_risk_records = _mapping(
        resolved.get("branch_risk_records"),
        field_name="branch_risk_records",
    )
    risk_evidence = _branch_risk_content_evidence(
        resolved,
        branch_risk_records,
        resolved_branches,
        semantic_routing_enabled=semantic_routing_enabled,
        null_space_enabled=null_space_enabled,
    )
    update_content = {}
    branch_update_fields = {
        "lf_update_content_sha256": "lf_content",
        "tail_robust_update_content_sha256": "tail_robust",
        "attention_geometry_update_content_sha256": "attention_geometry",
    }
    for field_name in _UPDATE_CONTENT_FIELDS:
        branch_name = branch_update_fields.get(field_name)
        if branch_name is not None and branch_name not in resolved_branches:
            if resolved.get(field_name) != "":
                raise ValueError("已禁用载体分支仍保留更新原子")
            update_content[field_name] = ""
        else:
            update_content[field_name] = _sha256(
                resolved.get(field_name),
                field_name=field_name,
            )
    update_content.update(
        {
            field_name: (
                _sha256(
                    resolved.get(field_name),
                    field_name=field_name,
                )
                if null_space_enabled
                else ""
            )
            for field_name in _QUANTIZED_WRITE_JACOBIAN_CONTENT_FIELDS
        }
    )
    recomputed_composition_digest = (
        recompute_quantized_composition_evidence_digest(resolved)
    )
    if recomputed_composition_digest != resolved.get(
        "quantized_composition_evidence_digest"
    ):
        raise ValueError("量化合成摘要不能由持久化叶子记录重建")
    if (
        resolved.get("quantized_write_original_latent_content_sha256")
        != resolved.get("latent_content_sha256_before")
        or resolved.get(
            "quantized_write_candidate_latent_content_sha256"
        )
        != resolved.get("latent_content_sha256_after")
    ):
        raise ValueError("量化写回的前后 latent 身份与注入原子不一致")

    null_space_records = _mapping(
        resolved.get("null_space_records"),
        field_name="null_space_records",
    )
    if set(null_space_records) != set(resolved_branches):
        raise ValueError("Null Space 记录顺序必须等于活动分支顺序")
    null_space_identities = []
    for branch_name in resolved_branches:
        subspace = _mapping(
            null_space_records[branch_name],
            field_name=f"null_space_records.{branch_name}",
        )
        if null_space_enabled:
            solver_digest = _sha256(
                subspace.get("solver_digest"),
                field_name=f"{branch_name}.solver_digest",
            )
            if recompute_jacobian_null_space_result_digest(
                subspace
            ) != solver_digest:
                raise ValueError("Null Space 摘要不能由八类内容身份重建")
            identity = {
                "branch_name": branch_name,
                "solver_digest": solver_digest,
                **{
                    field_name: _sha256(
                        subspace.get(field_name),
                        field_name=f"{branch_name}.{field_name}",
                    )
                    for field_name in _NULL_SPACE_CONTENT_FIELDS
                },
            }
        else:
            solver_role = subspace.get("solver")
            if not isinstance(solver_role, str) or not solver_role:
                raise ValueError("关闭 Null Space 时必须记录完整空间求解角色")
            identity = {
                "branch_name": branch_name,
                "solver_role": solver_role,
            }
        null_space_identities.append(
            {
                **identity,
                "null_space_record_digest": build_stable_digest(subspace),
            }
        )

    expected_attention = "attention_geometry" in resolved_branches
    if expected_attention:
        qk_records = resolved.get("attention_qk_atomic_content_records")
        qk_digest = _sha256(
            resolved.get("attention_qk_atomic_content_digest"),
            field_name="attention_qk_atomic_content_digest",
        )
        if qk_atomic_evaluation_records_digest(
            qk_records,
            "attention_qk_atomic_content_records",
        ) != qk_digest:
            raise ValueError("注入五角色 Q/K 摘要不能由逐层原子重建")
        roles = tuple(
            str(item.get("qk_evaluation_role", "")) for item in qk_records
        )
        if roles != (
            "latent_before",
            "optimization_content_base_latent",
            "accepted_attention_candidate",
            "actual_written_content_base_latent",
            "actual_written_combined_latent",
        ):
            raise ValueError("注入 Q/K 记录没有保留冻结五角色顺序")
        expected_layer_names = tuple(
            str(name) for name in resolved.get("attention_module_names", ())
        )
        if not qk_atomic_evaluation_records_ready(
            qk_records,
            qk_digest,
            aggregate_field_name="attention_qk_atomic_content_records",
            expected_roles=roles,
            expected_layer_names=expected_layer_names,
            require_evaluation_identity=True,
        ):
            raise ValueError("注入五角色 Q/K 逐层原子或评价身份不完整")
        operator_records = resolved.get(
            "attention_relation_qk_operator_metadata_records"
        )
        operator_digest = _sha256(
            resolved.get(
                "attention_relation_qk_operator_metadata_digest"
            ),
            field_name=(
                "attention_relation_qk_operator_metadata_digest"
            ),
        )
        if (
            qk_operator_metadata_records_digest(operator_records)
            != operator_digest
            or not qk_operator_metadata_records_ready(
                operator_records,
                expected_layer_names=expected_layer_names,
            )
        ):
            raise ValueError("注入 Q/K 精确算子元数据不能由逐层记录重建")
        attention_identity = {
            "attention_qk_atomic_content_digest": qk_digest,
            "attention_relation_qk_operator_metadata_digest": (
                operator_digest
            ),
            "attention_qk_evaluation_records_digest": build_stable_digest(
                qk_records
            ),
        }
    else:
        none_fields = (
            "attention_score_before",
            "attention_content_base_score",
            "attention_score_after",
            "attention_actual_written_content_base_score",
            "attention_final_combined_score",
            "attention_score_gain",
            "attention_applied_update_strength",
            "attention_backtracking_step_count",
        )
        empty_string_fields = (
            "attention_update_digest",
            "attention_update_content_sha256",
            "attention_update_unit_direction_content_sha256",
            "stable_token_selection_digest",
            "stable_pair_weight_identity_digest",
            "stable_pair_weight_realization_digest",
            "attention_relation_component_protocol_digest",
            "attention_relation_source",
            "attention_relation_component_identity_digest",
            "attention_relation_keyed_projection_digest",
            "attention_relation_qk_operator_metadata_digest",
            "attention_qk_atomic_content_digest",
        )
        empty_list_fields = (
            "stable_token_indices",
            "attention_relation_component_names",
            "attention_relation_active_component_names",
            "attention_relation_component_weights",
            "attention_relation_qk_operator_metadata_records",
            "attention_qk_atomic_content_records",
        )
        if (
            any(resolved.get(field_name) is not None for field_name in none_fields)
            or any(
                resolved.get(field_name) != ""
                for field_name in empty_string_fields
            )
            or any(
                resolved.get(field_name) != []
                for field_name in empty_list_fields
            )
            or resolved.get("attention_relation_direct_qk_source_ready") is not False
            or resolved.get("attention_relation_qk_operator_metadata_ready") is not False
            or resolved.get("attention_qk_atomic_content_ready") is not False
        ):
            raise ValueError("关闭 attention geometry 时仍保留注意力原子")
        attention_identity = {
            "attention_qk_atomic_content_digest": "",
            "attention_relation_qk_operator_metadata_digest": "",
            "attention_qk_evaluation_records_digest": "",
        }
    return {
        "step_index": int(resolved.get("step_index")),
        "scheduler_step_timestep": float(
            resolved.get("scheduler_step_timestep")
        ),
        "post_step_schedule_index": int(
            resolved.get("post_step_schedule_index")
        ),
        "watermark_key_material_digest_random": (
            watermark_key_material_digest_random
        ),
        "active_carrier_branches": resolved_branches,
        "semantic_routing_enabled": bool(semantic_routing_enabled),
        "null_space_enabled": bool(null_space_enabled),
        "risk_content_evidence": risk_evidence,
        "content_carrier_identity": {
            "lf_carrier_protocol_digest": lf_carrier_protocol_digest,
            "lf_template_content_sha256": lf_template_content_sha256,
            "lf_template_digest": lf_template_digest,
            "lf_template_shape": lf_template_shape,
            "lf_projection_energy_retention": (
                lf_projection_energy_retention
            ),
            "tail_carrier_protocol_digest": tail_carrier_protocol_digest,
            "tail_template_content_sha256": (
                tail_template_content_sha256
            ),
            "tail_template_digest": tail_template_digest,
            "tail_template_shape": tail_template_shape,
            "tail_template_element_count": tail_template_element_count,
            "tail_selected_element_count": tail_selected_element_count,
            "tail_projection_energy_retention": (
                tail_projection_energy_retention
            ),
            "tail_retained_fraction": resolved.get(
                "tail_retained_fraction"
            ),
        },
        "null_space_content_records": null_space_identities,
        "update_content_records": update_content,
        **attention_identity,
        "update_record_content_digest": build_stable_digest(resolved),
    }


def validate_scientific_update_content_identity(
    record: Mapping[str, Any],
    *,
    expected_branches: Sequence[str],
    null_space_enabled: bool,
    semantic_routing_enabled: bool,
) -> dict[str, Any]:
    """公开复验单个注入原子的活动科学算子身份."""

    return _update_content_identity(
        record,
        expected_branches=expected_branches,
        null_space_enabled=null_space_enabled,
        semantic_routing_enabled=semantic_routing_enabled,
    )


def _detection_content_identity(
    record: Mapping[str, Any],
    *,
    expected_attention: bool,
    expected_alignment: bool,
    expected_content_branches: Sequence[str],
    detection_index: int,
    detection_key_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """绑定一次盲检所评估图像、公开噪声和检测 Q/K 内容。"""

    resolved = dict(record)
    metadata = _mapping(
        resolved.get("metadata"),
        field_name="detection.metadata",
    )
    validate_image_only_detection_digest_record(resolved)
    detector_config_digest = _sha256(
        resolved.get("image_only_detector_config_digest"),
        field_name="detection.image_only_detector_config_digest",
    )
    if (
        metadata.get("attention_geometry_enabled") is not expected_attention
        or metadata.get("image_alignment_enabled") is not expected_alignment
    ):
        raise ValueError("盲检配置身份与正式方法机制开关不一致")
    if expected_alignment and not expected_attention:
        raise ValueError("图像 alignment 不能脱离 attention geometry 启用")
    detection_key_identity = validate_detection_key_identity_record(
        resolved,
        detection_key_plan,
    )
    watermark_key_material_digest_random = _sha256(
        resolved.get("watermark_key_material_digest_random"),
        field_name="detection.watermark_key_material_digest_random",
    )
    sample_role = str(resolved.get("sample_role", ""))
    attack_id = str(resolved.get("attack_id", "none"))
    attack_present = attack_id not in {"", "none"}
    detection_key_role = detection_key_identity["detection_key_role"]
    if (
        detection_key_role == REGISTERED_WATERMARK_KEY_ROLE
        and sample_role not in {"clean_negative", "positive_source"}
    ) or (
        detection_key_role == REGISTERED_WRONG_KEY_ROLE
        and (sample_role != "wrong_key_negative" or attack_present)
    ):
        raise ValueError("检测密钥角色与样本角色或攻击角色不一致")
    lf_carrier_protocol_digest = _sha256(
        resolved.get("lf_carrier_protocol_digest"),
        field_name="detection.lf_carrier_protocol_digest",
    )
    tail_carrier_protocol_digest = _sha256(
        resolved.get("tail_carrier_protocol_digest"),
        field_name="detection.tail_carrier_protocol_digest",
    )
    lf_weight = float(resolved.get("lf_weight"))
    tail_robust_weight = float(resolved.get("tail_robust_weight"))
    detected_content_branches = tuple(
        branch_name
        for branch_name, weight in (
            ("lf_content", lf_weight),
            ("tail_robust", tail_robust_weight),
        )
        if weight > 0.0
    )
    if detected_content_branches != tuple(expected_content_branches):
        raise ValueError("盲检活动内容分支与嵌入方法不一致")
    lf_template_content_sha256 = (
        _sha256(
            resolved.get("lf_template_content_sha256"),
            field_name="detection.lf_template_content_sha256",
        )
        if lf_weight > 0.0
        else ""
    )
    tail_template_content_sha256 = (
        _sha256(
            resolved.get("tail_template_content_sha256"),
            field_name="detection.tail_template_content_sha256",
        )
        if tail_robust_weight > 0.0
        else ""
    )
    tail_threshold = float(resolved.get("tail_threshold"))
    tail_retained_fraction = float(resolved.get("tail_retained_fraction"))
    tail_template_shape = list(resolved.get("tail_template_shape"))
    tail_template_element_count = int(
        resolved.get("tail_template_element_count")
    )
    tail_selected_element_count = int(
        resolved.get("tail_selected_element_count")
    )
    aligned_content_identity = {
        "aligned_lf_score": resolved.get("aligned_lf_score"),
        "aligned_tail_robust_score": resolved.get(
            "aligned_tail_robust_score"
        ),
        "aligned_content_score": resolved.get("aligned_content_score"),
    }
    attacked_digest = resolved.get("attacked_image_digest", "")
    if attacked_digest:
        attacked_digest = _sha256(
            attacked_digest,
            field_name="attacked_image_digest",
        )
    evaluated_pixels = _sha256(
        resolved.get("evaluated_image_rgb_uint8_content_sha256"),
        field_name="evaluated_image_rgb_uint8_content_sha256",
    )
    source_width = int(resolved.get("source_image_width"))
    source_height = int(resolved.get("source_image_height"))
    evaluated_width = int(resolved.get("evaluated_image_width"))
    evaluated_height = int(resolved.get("evaluated_image_height"))
    if min(
        source_width,
        source_height,
        evaluated_width,
        evaluated_height,
    ) <= 0:
        raise ValueError("检测图像尺寸必须为正整数")
    roles: tuple[str, ...] = ()
    if expected_attention:
        qk_records = metadata.get("detection_qk_atomic_content_records")
        qk_digest = _sha256(
            metadata.get("detection_qk_atomic_content_digest"),
            field_name="detection_qk_atomic_content_digest",
        )
        if qk_atomic_evaluation_records_digest(
            qk_records,
            "detection_qk_atomic_content_records",
        ) != qk_digest:
            raise ValueError("检测 Q/K 摘要不能由 raw/aligned 原子重建")
        roles = tuple(
            str(item.get("qk_evaluation_role", "")) for item in qk_records
        )
        expected_roles = (
            ("raw_detection_image", "aligned_detection_image")
            if expected_alignment
            else ("raw_detection_image",)
        )
        if roles != expected_roles:
            raise ValueError("检测 Q/K 没有保留冻结 raw/aligned 顺序")
        first_atoms = qk_records[0].get("qk_atomic_content_records")
        if not isinstance(first_atoms, list):
            raise ValueError("检测 Q/K 缺少逐层原子")
        expected_layer_names = tuple(
            str(item.get("record_layer_name", ""))
            for item in first_atoms
        )
        if not qk_atomic_evaluation_records_ready(
            qk_records,
            qk_digest,
            aggregate_field_name="detection_qk_atomic_content_records",
            expected_roles=roles,
            expected_layer_names=expected_layer_names,
        ):
            raise ValueError("检测 Q/K 逐层原子内容不完整")
        evidence_records = metadata.get(
            "public_detection_noise_evidence_records"
        )
        if not isinstance(evidence_records, list) or len(
            evidence_records
        ) != len(qk_records):
            raise ValueError("公开噪声记录与检测 Q/K 评价次数不一致")
        first_evaluation_index = evidence_records[0].get(
            "public_detection_noise_evaluation_index"
        )
        if (
            isinstance(first_evaluation_index, bool)
            or not isinstance(first_evaluation_index, int)
        ):
            raise ValueError("公开噪声评价索引必须为非负整数")
        public_noise_evidence = _public_noise_evidence_identity(
            evidence_records,
            metadata.get("public_detection_noise_evidence_digest"),
            aggregate_field_name=(
                "public_detection_noise_evidence_records"
            ),
            expected_indices=range(
                first_evaluation_index,
                first_evaluation_index + len(evidence_records),
            ),
        )
        content_digest = public_noise_evidence[
            "public_detection_noise_content_sha256"
        ]
        prg_digest = public_noise_evidence[
            "public_detection_noise_prg_identity_digest"
        ]
        if (
            metadata.get("public_detection_noise_content_sha256")
            != content_digest
            or metadata.get(
                "public_detection_noise_prg_identity_digest"
            )
            != prg_digest
        ):
            raise ValueError("公开噪声汇总身份与逐评价记录不一致")
        for qk_record, evidence_record in zip(
            qk_records,
            evidence_records,
        ):
            evidence_index = evidence_record.get(
                "public_detection_noise_evaluation_index"
            )
            if (
                qk_record.get(
                    "public_detection_noise_content_sha256"
                )
                != content_digest
                or qk_record.get(
                    "public_detection_noise_prg_identity_digest"
                )
                != prg_digest
                or qk_record.get(
                    "public_detection_noise_evaluation_index"
                )
                != evidence_record.get(
                    "public_detection_noise_evaluation_index"
                )
            ):
                raise ValueError("公开噪声没有逐评价绑定到检测 Q/K")
        image_qk_bindings = metadata.get(
            "detection_qk_image_content_bindings"
        )
        binding_digest = _sha256(
            metadata.get("detection_qk_image_content_binding_digest"),
            field_name="detection_qk_image_content_binding_digest",
        )
        if build_stable_digest(
            {"detection_qk_image_content_bindings": image_qk_bindings}
        ) != binding_digest:
            raise ValueError("检测图像与 Q/K 联合摘要不能重建")
        if not isinstance(image_qk_bindings, list) or len(
            image_qk_bindings
        ) != len(qk_records):
            raise ValueError("检测图像与 Q/K 绑定数量不一致")
        for binding, qk_record in zip(image_qk_bindings, qk_records):
            if (
                binding.get("qk_evaluation_role")
                != qk_record.get("qk_evaluation_role")
                or binding.get("qk_atomic_content_digest")
                != qk_record.get("qk_atomic_content_digest")
                or binding.get(
                    "public_detection_noise_evaluation_index"
                )
                != qk_record.get(
                    "public_detection_noise_evaluation_index"
                )
            ):
                raise ValueError("检测 Q/K 绑定与逐评价原子不一致")
        if image_qk_bindings[0].get(
            "evaluation_image_rgb_uint8_content_sha256"
        ) != evaluated_pixels:
            raise ValueError("raw 检测 Q/K 没有绑定实际评估图像")
        operator_records = metadata.get(
            "attention_relation_qk_operator_metadata_records"
        )
        operator_digest = _sha256(
            metadata.get(
                "attention_relation_qk_operator_metadata_digest"
            ),
            field_name="detection_qk_operator_metadata_digest",
        )
        if (
            qk_operator_metadata_records_digest(operator_records)
            != operator_digest
            or not qk_operator_metadata_records_ready(
                operator_records,
                expected_layer_names=expected_layer_names,
            )
        ):
            raise ValueError("检测 Q/K 精确算子元数据不能重建")
        public_noise_identity = {
            **public_noise_evidence,
            "detection_qk_atomic_content_digest": qk_digest,
            "detection_qk_image_content_binding_digest": binding_digest,
            "detection_qk_operator_metadata_digest": operator_digest,
        }
    else:
        empty_string_fields = (
            "public_detection_noise_content_sha256",
            "public_detection_noise_prg_identity_digest",
            "public_detection_noise_evidence_digest",
            "detection_qk_atomic_content_digest",
            "detection_qk_image_content_binding_digest",
            "attention_relation_qk_operator_metadata_digest",
        )
        empty_list_fields = (
            "public_detection_noise_evidence_records",
            "detection_qk_atomic_content_records",
            "detection_qk_image_content_bindings",
            "attention_relation_qk_operator_metadata_records",
        )
        if (
            resolved.get("raw_attention_geometry_score") is not None
            or any(metadata.get(name) not in {None, ""} for name in empty_string_fields)
            or any(
                metadata.get(name) not in (None, [])
                for name in empty_list_fields
            )
            or metadata.get("detection_qk_atomic_content_ready") is True
            or metadata.get("attention_relation_qk_operator_metadata_ready")
            is True
        ):
            raise ValueError("关闭 attention geometry 时仍保留检测 Q/K 原子")
        public_noise_identity = {
            "public_detection_noise_content_sha256": "",
            "public_detection_noise_prg_identity_digest": "",
            "public_detection_noise_evidence_digest": "",
            "public_detection_noise_evidence_records_digest": "",
            "detection_qk_atomic_content_digest": "",
            "detection_qk_image_content_binding_digest": "",
            "detection_qk_operator_metadata_digest": "",
            "public_detection_noise_evaluation_indices": [],
        }
    alignment_value = resolved.get("alignment")
    if alignment_value is None:
        alignment_digest = ""
        if expected_alignment:
            raise ValueError("aligned 检测 Q/K 缺少仿射恢复记录")
    else:
        if not expected_alignment:
            raise ValueError("关闭图像 alignment 时不得保留仿射恢复记录")
        alignment_record = _mapping(
            alignment_value,
            field_name="alignment",
        )
        alignment_digest = _sha256(
            alignment_record.get("alignment_digest"),
            field_name="alignment_digest",
        )
    return {
        "detection_index": detection_index,
        "sample_role": sample_role,
        "detection_key_identity": detection_key_identity,
        "watermark_key_material_digest_random": (
            watermark_key_material_digest_random
        ),
        "attack_id": attack_id,
        "source_image_path": str(resolved.get("source_image_path", "")),
        "source_image_file_sha256": _sha256(
            resolved.get("source_image_digest"),
            field_name="source_image_digest",
        ),
        "source_image_rgb_uint8_content_sha256": _sha256(
            resolved.get("source_image_rgb_uint8_content_sha256"),
            field_name="source_image_rgb_uint8_content_sha256",
        ),
        "source_image_width": source_width,
        "source_image_height": source_height,
        "evaluated_image_path": str(
            resolved.get("evaluated_image_path", "")
        ),
        "evaluated_image_file_sha256": _sha256(
            resolved.get("evaluated_image_digest"),
            field_name="evaluated_image_digest",
        ),
        "evaluated_image_rgb_uint8_content_sha256": evaluated_pixels,
        "content_carrier_identity": {
            "lf_carrier_protocol_digest": lf_carrier_protocol_digest,
            "lf_template_content_sha256": lf_template_content_sha256,
            "tail_carrier_protocol_digest": tail_carrier_protocol_digest,
            "tail_template_content_sha256": tail_template_content_sha256,
            "tail_template_shape": tail_template_shape,
            "tail_template_element_count": tail_template_element_count,
            "tail_selected_element_count": tail_selected_element_count,
            "tail_threshold": tail_threshold,
            "tail_retained_fraction": tail_retained_fraction,
            "lf_weight": lf_weight,
            "tail_robust_weight": tail_robust_weight,
            "lf_score": resolved.get("lf_score"),
            "tail_robust_score": resolved.get("tail_robust_score"),
            "content_score": resolved.get("content_score"),
            **aligned_content_identity,
        },
        "raw_attention_geometry_score": resolved.get(
            "raw_attention_geometry_score"
        ),
        "image_only_detector_config_digest": detector_config_digest,
        "detector_digest": _sha256(
            resolved.get("detector_digest"),
            field_name="detector_digest",
        ),
        "evaluated_image_width": evaluated_width,
        "evaluated_image_height": evaluated_height,
        "attacked_image_digest": attacked_digest,
        "alignment_digest": alignment_digest,
        **public_noise_identity,
        "detection_record_content_digest": build_stable_digest(resolved),
    }


def build_scientific_content_binding_record(
    *,
    run_id: str,
    method_definition_digest: str,
    scientific_unit_config_digest: str,
    full_update_records: Sequence[Mapping[str, Any]],
    carrier_only_update_records: Sequence[Mapping[str, Any]],
    detection_records: Sequence[Mapping[str, Any]],
    detection_key_plan: Mapping[str, Any],
    final_image_records: Mapping[str, Mapping[str, Any]],
    final_image_attention_observability: Mapping[str, Any] | None,
    final_image_preservation: Mapping[str, Any] | None,
    carrier_only_final_image_preservation: Mapping[str, Any] | None,
    carrier_only_counterfactual: Mapping[str, Any] | None,
    attention_geometry_enabled: bool,
    image_alignment_enabled: bool,
    semantic_routing_enabled: bool,
    null_space_enabled: bool,
    full_active_branches: Sequence[str],
    carrier_only_active_branches: Sequence[str],
) -> dict[str, Any]:
    """构造可由持久化叶子记录和图像文件完整重建的总证据记录。"""

    if not run_id:
        raise ValueError("scientific content binding 缺少 run_id")
    normalized_full_branches = tuple(
        name for name in _BRANCH_NAMES if name in full_active_branches
    )
    normalized_carrier_branches = tuple(
        name for name in _BRANCH_NAMES if name in carrier_only_active_branches
    )
    if (
        tuple(full_active_branches) != normalized_full_branches
        or len(set(full_active_branches)) != len(tuple(full_active_branches))
        or not normalized_full_branches
        or tuple(carrier_only_active_branches) != normalized_carrier_branches
        or len(set(carrier_only_active_branches))
        != len(tuple(carrier_only_active_branches))
        or image_alignment_enabled and not attention_geometry_enabled
        or ("attention_geometry" in normalized_full_branches)
        is not attention_geometry_enabled
        or (
            attention_geometry_enabled
            and normalized_carrier_branches
            != tuple(
                name
                for name in normalized_full_branches
                if name != "attention_geometry"
            )
        )
        or (not attention_geometry_enabled and normalized_carrier_branches)
    ):
        raise ValueError("科学内容绑定的机制开关或活动分支集合无效")
    full_identities = [
        _update_content_identity(
            record,
            expected_branches=full_active_branches,
            null_space_enabled=null_space_enabled,
            semantic_routing_enabled=semantic_routing_enabled,
        )
        for record in full_update_records
    ]
    if not full_identities or len(
        {item["step_index"] for item in full_identities}
    ) != len(full_identities):
        raise ValueError("完整方法注入步骤必须非空且不得重复")
    carrier_identities = [
        _update_content_identity(
            record,
            expected_branches=carrier_only_active_branches,
            null_space_enabled=null_space_enabled,
            semantic_routing_enabled=semantic_routing_enabled,
        )
        for record in carrier_only_update_records
    ]
    if attention_geometry_enabled and (
        len(carrier_identities) != len(full_identities)
        or [item["step_index"] for item in carrier_identities]
        != [item["step_index"] for item in full_identities]
    ):
        raise ValueError("carrier-only 与完整方法的注入步骤身份不一致")
    if not attention_geometry_enabled and carrier_identities:
        raise ValueError("关闭 attention geometry 时不得存在 carrier-only 轨迹")
    validated_detection_key_plan = validate_detection_key_plan_record(
        detection_key_plan
    )
    detection_identities = [
        _detection_content_identity(
            record,
            expected_attention=attention_geometry_enabled,
            expected_alignment=image_alignment_enabled,
            expected_content_branches=tuple(
                branch_name
                for branch_name in normalized_full_branches
                if branch_name in {"lf_content", "tail_robust"}
            ),
            detection_index=index,
            detection_key_plan=validated_detection_key_plan,
        )
        for index, record in enumerate(detection_records)
    ]
    if not detection_identities:
        raise ValueError("总科学内容绑定缺少仅图像检测记录")
    detector_config_digests = {
        identity["image_only_detector_config_digest"]
        for identity in detection_identities
    }
    if len(detector_config_digests) != 1:
        raise ValueError("同一科学单元的检测记录混用了不同盲检配置")
    update_key_digests = {
        identity["watermark_key_material_digest_random"]
        for identity in (*full_identities, *carrier_identities)
    }
    if update_key_digests != {
        validated_detection_key_plan[
            "registered_watermark_key_digest_random"
        ]
    }:
        raise ValueError("注入轨迹与检测密钥计划未共享同一注册密钥身份")
    registered_key_detections = [
        identity
        for identity in detection_identities
        if identity["detection_key_identity"]["detection_key_role"]
        == REGISTERED_WATERMARK_KEY_ROLE
    ]
    wrong_key_detections = [
        identity
        for identity in detection_identities
        if identity["detection_key_identity"]["detection_key_role"]
        == REGISTERED_WRONG_KEY_ROLE
    ]
    if not registered_key_detections or not wrong_key_detections:
        raise ValueError("检测记录必须同时覆盖注册密钥与预注册 wrong-key 角色")
    content_carrier_cross_path_identity: dict[str, dict[str, str]] = {}
    for (
        branch_name,
        template_field_name,
        protocol_field_name,
        template_shape_field_name,
    ) in (
        (
            "lf_content",
            "lf_template_content_sha256",
            "lf_carrier_protocol_digest",
            "lf_template_shape",
        ),
        (
            "tail_robust",
            "tail_template_content_sha256",
            "tail_carrier_protocol_digest",
            "tail_template_shape",
        ),
    ):
        update_template_digests = {
            identity["content_carrier_identity"][template_field_name]
            for identity in (*full_identities, *carrier_identities)
            if branch_name in identity["active_carrier_branches"]
        }
        update_protocol_digests = {
            identity["content_carrier_identity"][protocol_field_name]
            for identity in (*full_identities, *carrier_identities)
            if branch_name in identity["active_carrier_branches"]
        }
        update_template_shapes = {
            tuple(
                identity["content_carrier_identity"][
                    template_shape_field_name
                ]
            )
            for identity in (*full_identities, *carrier_identities)
            if branch_name in identity["active_carrier_branches"]
        }
        if not update_template_digests:
            continue
        registered_template_digests = {
            identity["content_carrier_identity"][template_field_name]
            for identity in registered_key_detections
        }
        wrong_key_template_digests = {
            identity["content_carrier_identity"][template_field_name]
            for identity in wrong_key_detections
        }
        detection_protocol_digests = {
            identity["content_carrier_identity"][protocol_field_name]
            for identity in detection_identities
        }
        detection_template_shapes = (
            {
                tuple(
                    identity["content_carrier_identity"][
                        "tail_template_shape"
                    ]
                )
                for identity in detection_identities
            }
            if branch_name == "tail_robust"
            else update_template_shapes
        )
        if (
            len(update_template_digests) != 1
            or registered_template_digests != update_template_digests
            or len(wrong_key_template_digests) != 1
            or wrong_key_template_digests == update_template_digests
            or len(update_protocol_digests) != 1
            or detection_protocol_digests != update_protocol_digests
            or len(update_template_shapes) != 1
            or detection_template_shapes != update_template_shapes
        ):
            raise ValueError(
                f"{branch_name} 在嵌入与仅图像检测路径中的固定模板身份不一致"
            )
        content_carrier_cross_path_identity[branch_name] = {
            "registered_template_content_sha256": next(
                iter(update_template_digests)
            ),
            "wrong_key_template_content_sha256": next(
                iter(wrong_key_template_digests)
            ),
            "carrier_protocol_digest": next(
                iter(update_protocol_digests)
            ),
            "template_shape": list(next(iter(update_template_shapes))),
        }
    if attention_geometry_enabled:
        public_noise_indices = [
            evaluation_index
            for identity in detection_identities
            for evaluation_index in identity[
                "public_detection_noise_evaluation_indices"
            ]
        ]
        if (
            not public_noise_indices
            or public_noise_indices
            != list(range(3, 3 + len(public_noise_indices)))
            or len(
                {
                    identity[
                        "public_detection_noise_content_sha256"
                    ]
                    for identity in detection_identities
                }
            )
            != 1
            or len(
                {
                    identity[
                        "public_detection_noise_prg_identity_digest"
                    ]
                    for identity in detection_identities
                }
            )
            != 1
        ):
            raise ValueError(
                "检测记录必须共享同一公开噪声, 且连续全局索引必须紧接最终三图评价"
            )

    expected_image_roles = (
        ("clean_image", "carrier_only_image", "watermarked_image")
        if attention_geometry_enabled
        else ("clean_image", "watermarked_image")
    )
    if tuple(final_image_records) != expected_image_roles:
        raise ValueError("最终图像角色顺序与方法配置不一致")
    final_images = []
    for role in expected_image_roles:
        image_record = _mapping(
            final_image_records[role],
            field_name=f"final_image_records.{role}",
        )
        if image_record.get("image_rgb_uint8_content_schema") != (
            IMAGE_RGB_UINT8_CONTENT_SCHEMA
        ):
            raise ValueError("最终图像使用了错误的规范 RGB 内容版本")
        image_width = int(image_record.get("image_width"))
        image_height = int(image_record.get("image_height"))
        if image_width <= 0 or image_height <= 0:
            raise ValueError("最终图像尺寸必须为正整数")
        final_images.append(
            {
                "image_role": role,
                "image_path": str(image_record.get("image_path", "")),
                "image_file_sha256": _sha256(
                    image_record.get("image_file_sha256"),
                    field_name=f"{role}.image_file_sha256",
                ),
                "image_rgb_uint8_content_schema": (
                    IMAGE_RGB_UINT8_CONTENT_SCHEMA
                ),
                "image_rgb_uint8_content_sha256": _sha256(
                    image_record.get(
                        "image_rgb_uint8_content_sha256"
                    ),
                    field_name=(
                        f"{role}.image_rgb_uint8_content_sha256"
                    ),
                ),
                "image_width": image_width,
                "image_height": image_height,
            }
        )

    observability = dict(final_image_attention_observability or {})
    if attention_geometry_enabled:
        final_qk_digest = _sha256(
            observability.get("final_image_qk_atomic_content_digest"),
            field_name="final_image_qk_atomic_content_digest",
        )
        qk_records = observability.get(
            "final_image_qk_atomic_content_records"
        )
        if qk_atomic_evaluation_records_digest(
            qk_records,
            "final_image_qk_atomic_content_records",
        ) != final_qk_digest:
            raise ValueError("最终三图 Q/K 摘要不能由逐层原子重建")
        final_qk_roles = (
            "final_clean_image",
            "final_carrier_only_image",
            "final_watermarked_image",
        )
        final_qk_layer_names = tuple(
            str(name)
            for name in observability.get("attention_module_names", ())
        )
        if not qk_atomic_evaluation_records_ready(
            qk_records,
            final_qk_digest,
            aggregate_field_name="final_image_qk_atomic_content_records",
            expected_roles=final_qk_roles,
            expected_layer_names=final_qk_layer_names,
        ):
            raise ValueError("最终三图 Q/K 逐层原子内容不完整")
        final_public_noise_evidence = _public_noise_evidence_identity(
            observability.get(
                "final_image_public_detection_noise_evidence_records"
            ),
            observability.get(
                "final_image_public_detection_noise_evidence_digest"
            ),
            aggregate_field_name=(
                "final_image_public_detection_noise_evidence_records"
            ),
            expected_indices=(0, 1, 2),
        )
        if (
            observability.get(
                "final_image_public_detection_noise_content_sha256"
            )
            != final_public_noise_evidence[
                "public_detection_noise_content_sha256"
            ]
            or observability.get(
                "final_image_public_detection_noise_prg_identity_digest"
            )
            != final_public_noise_evidence[
                "public_detection_noise_prg_identity_digest"
            ]
            or observability.get(
                "final_image_public_detection_noise_evidence_ready"
            )
            is not True
        ):
            raise ValueError("最终三图公开噪声汇总身份不一致")
        for qk_record, evaluation_index in zip(
            qk_records,
            (0, 1, 2),
        ):
            if (
                qk_record.get(
                    "public_detection_noise_content_sha256"
                )
                != final_public_noise_evidence[
                    "public_detection_noise_content_sha256"
                ]
                or qk_record.get(
                    "public_detection_noise_prg_identity_digest"
                )
                != final_public_noise_evidence[
                    "public_detection_noise_prg_identity_digest"
                ]
                or qk_record.get(
                    "public_detection_noise_evaluation_index"
                )
                != evaluation_index
            ):
                raise ValueError("最终三图 Q/K 没有逐角色绑定公开噪声")
        qk_bindings = observability.get(
            "final_image_qk_image_content_bindings"
        )
        qk_binding_digest = _sha256(
            observability.get(
                "final_image_qk_image_content_binding_digest"
            ),
            field_name="final_image_qk_image_content_binding_digest",
        )
        if build_stable_digest(
            {"final_image_qk_image_content_bindings": qk_bindings}
        ) != qk_binding_digest:
            raise ValueError("最终图像与 Q/K 联合摘要不能重建")
        if not isinstance(qk_bindings, list) or tuple(
            item.get("qk_evaluation_role") for item in qk_bindings
        ) != final_qk_roles:
            raise ValueError("最终图像与 Q/K 绑定缺少冻结三角色")
        for qk_record, qk_binding, image_record in zip(
            qk_records,
            qk_bindings,
            final_images,
        ):
            if (
                qk_binding.get("qk_evaluation_role")
                != qk_record.get("qk_evaluation_role")
                or qk_binding.get("qk_atomic_content_digest")
                != qk_record.get("qk_atomic_content_digest")
                or qk_binding.get(
                    "evaluation_image_rgb_uint8_content_sha256"
                )
                != image_record["image_rgb_uint8_content_sha256"]
                or qk_binding.get(
                    "public_detection_noise_content_sha256"
                )
                != qk_record.get(
                    "public_detection_noise_content_sha256"
                )
                or qk_binding.get(
                    "public_detection_noise_prg_identity_digest"
                )
                != qk_record.get(
                    "public_detection_noise_prg_identity_digest"
                )
                or qk_binding.get(
                    "public_detection_noise_evaluation_index"
                )
                != qk_record.get(
                    "public_detection_noise_evaluation_index"
                )
            ):
                raise ValueError("最终图像像素身份没有逐角色绑定 Q/K 原子")
        final_qk_operator_digest = _sha256(
            observability.get(
                "attention_relation_qk_operator_metadata_digest"
            ),
            field_name="final_image_qk_operator_metadata_digest",
        )
        final_operator_records = observability.get(
            "attention_relation_qk_operator_metadata_records"
        )
        if (
            qk_operator_metadata_records_digest(final_operator_records)
            != final_qk_operator_digest
            or not qk_operator_metadata_records_ready(
                final_operator_records,
                expected_layer_names=final_qk_layer_names,
            )
        ):
            raise ValueError("最终图像 Q/K 精确算子元数据不能重建")
    else:
        final_qk_digest = ""
        qk_binding_digest = ""
        final_qk_operator_digest = ""
        final_public_noise_evidence = {
            "public_detection_noise_content_sha256": "",
            "public_detection_noise_prg_identity_digest": "",
            "public_detection_noise_evidence_digest": "",
            "public_detection_noise_evidence_records_digest": "",
            "public_detection_noise_evaluation_indices": [],
        }
    if attention_geometry_enabled and (
        detection_identities[0][
            "public_detection_noise_content_sha256"
        ]
        != final_public_noise_evidence[
            "public_detection_noise_content_sha256"
        ]
        or detection_identities[0][
            "public_detection_noise_prg_identity_digest"
        ]
        != final_public_noise_evidence[
            "public_detection_noise_prg_identity_digest"
        ]
    ):
        raise ValueError("最终三图与后续检测没有共享同一公开噪声身份")

    payload = {
        "scientific_content_binding_schema": (
            SCIENTIFIC_CONTENT_BINDING_SCHEMA
        ),
        "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
        "image_rgb_uint8_content_schema": IMAGE_RGB_UINT8_CONTENT_SCHEMA,
        "run_id": run_id,
        "method_definition_digest": _sha256(
            method_definition_digest,
            field_name="method_definition_digest",
        ),
        "scientific_unit_config_digest": _sha256(
            scientific_unit_config_digest,
            field_name="scientific_unit_config_digest",
        ),
        "attention_geometry_enabled": bool(attention_geometry_enabled),
        "image_alignment_enabled": bool(image_alignment_enabled),
        "semantic_routing_enabled": bool(semantic_routing_enabled),
        "null_space_enabled": bool(null_space_enabled),
        "full_active_branches": list(full_active_branches),
        "carrier_only_active_branches": list(carrier_only_active_branches),
        "full_update_content_identities": full_identities,
        "full_update_content_bundle_digest": build_stable_digest(
            full_identities
        ),
        "carrier_only_update_content_identities": carrier_identities,
        "carrier_only_update_content_bundle_digest": build_stable_digest(
            carrier_identities
        ),
        "detection_content_identities": detection_identities,
        "image_only_detector_config_digest": next(
            iter(detector_config_digests)
        ),
        "detection_content_bundle_digest": build_stable_digest(
            detection_identities
        ),
        "detection_key_plan_digest_random": validated_detection_key_plan[
            "detection_key_plan_digest_random"
        ],
        "content_carrier_cross_path_identity": (
            content_carrier_cross_path_identity
        ),
        "final_image_content_records": final_images,
        "final_image_content_bundle_digest": build_stable_digest(
            final_images
        ),
        "final_image_qk_atomic_content_digest": final_qk_digest,
        "final_image_qk_image_content_binding_digest": qk_binding_digest,
        "final_image_qk_operator_metadata_digest": (
            final_qk_operator_digest
        ),
        "final_image_public_detection_noise_identity": (
            final_public_noise_evidence
        ),
        "final_image_attention_observability_record_digest": (
            build_stable_digest(observability)
        ),
        "final_image_preservation_record_digest": build_stable_digest(
            dict(final_image_preservation or {})
        ),
        "carrier_only_final_image_preservation_record_digest": (
            build_stable_digest(
                dict(carrier_only_final_image_preservation or {})
            )
        ),
        "carrier_only_counterfactual_record_digest": build_stable_digest(
            dict(carrier_only_counterfactual or {})
        ),
    }
    return {
        **payload,
        "scientific_content_binding_digest": build_stable_digest(payload),
    }


def recompute_scientific_content_binding_digest(
    record: Mapping[str, Any],
) -> str:
    """从持久化总证据记录重算顶层科学内容摘要。"""

    resolved = dict(record)
    if resolved.get("scientific_content_binding_schema") != (
        SCIENTIFIC_CONTENT_BINDING_SCHEMA
    ):
        raise ValueError("scientific content binding schema 不受支持")
    supplied = resolved.pop("scientific_content_binding_digest", None)
    _sha256(supplied, field_name="scientific_content_binding_digest")
    return build_stable_digest(resolved)


__all__ = [
    "IMAGE_RGB_UINT8_CONTENT_SCHEMA",
    "SCIENTIFIC_CONTENT_BINDING_SCHEMA",
    "build_scientific_content_binding_record",
    "canonical_rgb_uint8_content_record",
    "read_canonical_rgb_uint8_content_record",
    "recompute_scientific_content_binding_digest",
    "validate_scientific_update_content_identity",
]
