"""验证真实风险、Jacobian、载体、注意力和仅图像检测算子。"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch
from PIL import Image
import main.core.keyed_prg as keyed_prg_module
import main.methods.carrier.keyed_tensor as keyed_tensor_module
import main.methods.geometry.differentiable_attention as attention_module
import main.methods.subspace.jacobian_nullspace as nullspace_module

from main.methods.carrier import (
    KEYED_PRG_VERSION,
    build_low_frequency_template,
    build_tail_robust_template,
    compute_blind_content_score,
    keyed_prg_protocol_record,
)
from main.methods.detection import ImageOnlyDetectionConfig, detect_image_only_watermark
from main.methods.geometry import (
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    ATTENTION_RELATION_COMPONENT_NAMES,
    DifferentiableAttentionRecorder,
    QKAttentionRelation,
    attention_geometry_score,
    attention_relation_component_protocol,
    attention_relation_component_scores,
    attention_relation_stability_map,
    build_attention_relation_descriptor,
    build_attention_relation_graph_identity,
    build_qk_atomic_content_metadata,
    build_stable_attention_pair_weights,
    combine_attention_relation_component_scores,
    keyed_attention_relation_projection,
    optimize_attention_geometry_update,
    qk_atomic_content_records_digest,
    qk_self_attention,
    recover_attention_affine_alignment,
    select_stable_attention_tokens,
)
from main.methods.geometry.differentiable_attention import keyed_relation_signs
from main.methods.semantic import build_branch_risk_fields
from main.methods.subspace import (
    JacobianNullSpaceResult,
    build_exact_jacobian_linearization,
    exact_jvp,
    generate_keyed_candidate_directions,
    solve_jacobian_null_space,
    solve_psd_conjugate_gradient,
)
from experiments.runners.image_only_dataset_runtime import (
    FrozenEvidenceProtocol,
    _scientific_update_record_ready,
    apply_frozen_evidence_protocol,
    calibrate_complete_evidence_protocol,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    _align_image,
    _attention_modules,
    _image_attention_extractor,
    _public_detection_noise_seed,
    build_semantic_watermark_run_id,
    load_completed_semantic_watermark_runtime_result,
    semantic_watermark_runtime_config_payload,
)
from experiments.runtime.repository_environment import resolve_code_version
from experiments.runtime.diffusion.semantic_features import (
    JOINT_FEATURE_WIDTH,
    SEMANTIC_FEATURE_SCHEMA,
    SEMANTIC_FEATURE_WIDTH,
    HANDCRAFTED_STRUCTURE_FEATURE_SCHEMA,
    HANDCRAFTED_STRUCTURE_FEATURE_WIDTH,
)
from main.core.digest import build_stable_digest
from main.core.digest import (
    TENSOR_CONTENT_DIGEST_VERSION,
    tensor_content_sha256,
)
from scripts import semantic_watermark_scientific_workflow as scientific_workflow
from tests.helpers.scientific_unit_provenance import (
    build_test_scientific_unit_provenance,
)


def _direct_qk_relation_from_logits(
    logits: torch.Tensor,
    layer_name: str = "test_qk_relation_layer",
) -> QKAttentionRelation:
    """由显式 Q/K logits 构造生产路径相同的双张量关系对象。"""

    resolved = logits.unsqueeze(0) if logits.ndim == 2 else logits
    token_count = int(resolved.shape[-1])
    grid_side = int(round(token_count**0.5))
    centered_logits = resolved - resolved.mean(dim=-1, keepdim=True)
    probabilities = torch.softmax(resolved, dim=-1)
    return QKAttentionRelation(
        centered_logits=centered_logits,
        probabilities=probabilities,
        metadata={
            "module_layer_name": layer_name,
            "module_class_name": "tests.DirectQKRelation",
            "head_count": 1,
            "head_width": 1,
            "attention_scale": 1.0,
            "attention_scale_source": "inverse_sqrt_head_width",
            "q_normalization_applied": False,
            "k_normalization_applied": False,
            "q_normalization_class": "",
            "k_normalization_class": "",
            "source_token_count": token_count,
            "source_grid_side": grid_side,
            "sampled_token_count": token_count,
            "sampled_grid_side": grid_side,
            "sampled_token_indices": list(range(token_count)),
            "coordinate_convention": ATTENTION_COORDINATE_CONVENTION,
            "grid_align_corners": ATTENTION_GRID_ALIGN_CORNERS,
            **build_qk_atomic_content_metadata(
                layer_name,
                resolved,
                resolved,
                centered_logits,
                probabilities,
                tuple(range(token_count)),
            ),
            "centered_logit_aggregation": (
                "mean_of_per_head_row_centered_sampled_qk_logits"
            ),
            "relation_probability_aggregation": (
                "mean_of_per_head_sampled_image_token_probabilities"
            ),
            "mean_probability_is_softmax_of_mean_logits": False,
        },
    )


def write_recovery_candidate(
    root_path: Path,
    role: str,
    *,
    suffix: str = "20260712t000000z",
    generated_at_utc: datetime | None = None,
) -> SimpleNamespace:
    """构造闭合包恢复选择测试使用的最小候选对象."""

    destination_dir = root_path / "drive" / role
    destination_dir.mkdir(parents=True, exist_ok=True)
    package_path = destination_dir / f"{role}_package_{suffix}.zip"
    output_role = (
        "formal_mechanism_ablation"
        if role == "runtime_rerun_ablation"
        else role
    )
    with ZipFile(package_path, "w") as archive:
        archive.writestr(
            f"outputs/{output_role}/probe_paper/recovered_{role}.json",
            f"{role}:{suffix}",
        )
    return SimpleNamespace(
        package_path=package_path,
        package_sha256=scientific_workflow.file_sha256(package_path),
        generated_at_utc=(
            generated_at_utc
            or datetime(2026, 7, 12, tzinfo=timezone.utc)
        ),
        generated_at="2026-07-12T00:00:00+00:00",
        code_version="a" * 40,
        formal_execution_run_lock_digest="b" * 64,
        formal_execution_package_lock_digest="b" * 64,
        scientific_profile_id=scientific_workflow.SCIENTIFIC_PROFILE_ID,
        scientific_profile_digest="c" * 64,
        scientific_direct_requirements_digest="d" * 64,
        scientific_complete_hash_lock_digest="e" * 64,
        scientific_complete_hash_lock_dependency_count=17,
    )


@pytest.mark.quick
def test_image_only_attention_noise_seed_does_not_depend_on_generation_seed_or_prompt() -> None:
    """盲检公开噪声不得依赖生成种子、Prompt 或样本序号。"""

    base = SemanticWatermarkRuntimeConfig()
    changed_sample = replace(base, seed=base.seed + 999, prompt="完全不同的生成条件", prompt_id="other")
    changed_model = SimpleNamespace(
        injection_step_indices=base.injection_step_indices,
        carrier_model_reference=(
            "Manojb/stable-diffusion-2-1-base@"
            "0094d483a120f3f33dafbd187ea4aa60d10de75c"
        ),
        width=base.width,
        height=base.height,
        inference_steps=base.inference_steps,
    )

    assert _public_detection_noise_seed(base) == _public_detection_noise_seed(changed_sample)
    assert _public_detection_noise_seed(base) != _public_detection_noise_seed(changed_model)


@pytest.mark.quick
def test_branch_risk_fields_use_opposite_texture_preferences() -> None:
    """LF 应回避高纹理, 尾部鲁棒分支应偏好高纹理。"""

    fields = build_branch_risk_fields(
        semantic_values=(0.2, 0.2),
        texture_values=(0.1, 0.9),
        adjacent_step_stability_values=(0.8, 0.8),
        local_contrast_risk_values=(0.2, 0.2),
        attention_stability_values=(0.8, 0.8),
    )

    assert fields.lf_content.risk_values[0] < fields.lf_content.risk_values[1]
    assert fields.tail_robust.risk_values[0] > fields.tail_robust.risk_values[1]
    assert fields.lf_content.risk_field_digest != fields.tail_robust.risk_field_digest
    assert len(fields.lf_content.risk_values_content_sha256) == 64
    assert len(fields.lf_content.budget_values_content_sha256) == 64
    assert len(fields.lf_content.eligible_mask_content_sha256) == 64


@pytest.mark.quick
def test_full_jacobian_constraint_projection_recovers_null_direction() -> None:
    """JVP/VJP 约束投影应恢复完整特征不响应的 latent 方向。"""

    latent = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    def full_features(values: torch.Tensor) -> torch.Tensor:
        return torch.stack((values[0] ** 2, 3.0 * values[1], 2.0 * values[2]))

    _, tangent = exact_jvp(full_features, latent, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    linearization = build_exact_jacobian_linearization(full_features, latent)
    result = solve_jacobian_null_space(
        latent=latent,
        candidate_matrix=torch.eye(4)[:, (3, 0, 1, 2)],
        risk_budget=torch.ones_like(latent),
        null_rank=1,
        joint_feature_linearization=linearization,
        branch_name="lf_content",
    )

    assert tangent.tolist() == pytest.approx([2.0, 0.0, 0.0])
    assert result.response_residual == pytest.approx(0.0, abs=1e-7)
    assert result.orthogonality_error == pytest.approx(0.0, abs=1e-6)
    assert result.latent_basis.requires_grad is False
    assert result.response_matrix.requires_grad is False
    assert abs(float(result.latent_basis[3, 0])) == pytest.approx(1.0, abs=1e-6)
    assert result.metadata["solver"] == "matrix_free_full_jacobian_psd_cg"
    assert result.metadata["cg_damping"] == 0.0
    assert result.to_record()["latent_basis_content_sha256"] == (
        tensor_content_sha256(result.latent_basis)
    )
    assert result.to_record()["response_matrix_content_sha256"] == (
        tensor_content_sha256(result.response_matrix)
    )


@pytest.mark.quick
def test_risk_budget_is_explicit_in_full_jacobian_null_projection() -> None:
    """风险预算的零支持必须在完整 Jacobian Null Space 基底中保持为零。"""

    latent = torch.zeros(8)
    jacobian = torch.tensor(
        (
            (1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0),
        )
    )

    def full_features(values: torch.Tensor) -> torch.Tensor:
        return jacobian @ values

    linearization = build_exact_jacobian_linearization(full_features, latent)
    candidates = generate_keyed_candidate_directions(
        latent,
        "full_jacobian_key",
        "lf_content",
        candidate_count=8,
        preferred_directions=(torch.ones_like(latent),),
    )
    risk_budget = torch.tensor((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0))
    result = solve_jacobian_null_space(
        latent=latent,
        candidate_matrix=candidates,
        risk_budget=risk_budget,
        null_rank=4,
        joint_feature_linearization=linearization,
        branch_name="lf_content",
    )

    assert result.basis_rank == 4
    assert max(result.column_relative_response_residuals) <= 1e-4
    assert torch.linalg.norm(jacobian @ result.latent_basis).item() <= 1e-5
    assert torch.linalg.norm(result.latent_basis[-1]).item() == pytest.approx(0.0, abs=1e-7)
    assert all(value >= 0.01 for value in result.projection_energy_retentions)


@pytest.mark.quick
def test_undamped_psd_cg_reports_non_convergence_without_fallback() -> None:
    """迭代预算不足时 PSD-CG 必须报告失败, 不能加入阻尼后继续。"""

    diagonal = torch.tensor((1.0, 4.0))
    result = solve_psd_conjugate_gradient(
        lambda value: diagonal * value,
        torch.ones(2),
        maximum_iterations=1,
        relative_tolerance=1e-8,
    )

    assert result.converged is False
    assert result.iteration_count == 1
    assert result.relative_residual > 1e-8


@pytest.mark.quick
def test_candidate_matrix_preserves_preferred_carrier_direction() -> None:
    """候选矩阵必须显式包含固定载体, 避免随机低秩子空间丢失盲检能量。"""

    latent = torch.zeros(1, 1, 2, 2)
    preferred = torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]])

    candidates = generate_keyed_candidate_directions(
        latent,
        "preferred_key",
        "lf_content",
        candidate_count=3,
        preferred_directions=(preferred,),
    )

    first_direction = candidates[:, 0]
    assert abs(float(first_direction[-1])) == pytest.approx(1.0, abs=1e-6)
    assert torch.linalg.norm(first_direction[:-1]).item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.quick
def test_scientific_operator_gate_requires_all_real_operator_evidence() -> None:
    """关键算子门禁必须同时检查 JVP、残差、载体能量和 Q/K 提升。"""

    subspace = {
        "response_residual": 0.1,
        "relative_response_residual": 1e-6,
        "orthogonality_error": 1e-6,
        "column_relative_response_residuals": [1e-6] * 4,
        "projection_energy_retentions": [0.2] * 4,
        "cg_relative_residuals": [1e-7] * 4,
        "cg_converged": True,
        "metadata": {
            "jvp_mode": "torch_func_exact_jvp_vjp",
            "solver": "matrix_free_full_jacobian_psd_cg",
            "preferred_direction_count": 1,
            "semantic_feature_schema": SEMANTIC_FEATURE_SCHEMA,
            "semantic_feature_width": SEMANTIC_FEATURE_WIDTH,
            "handcrafted_structure_feature_schema": HANDCRAFTED_STRUCTURE_FEATURE_SCHEMA,
            "handcrafted_structure_feature_width": HANDCRAFTED_STRUCTURE_FEATURE_WIDTH,
            "joint_feature_width": JOINT_FEATURE_WIDTH,
            "feature_compression_applied": False,
            "keyed_prg_version": KEYED_PRG_VERSION,
            "keyed_prg_protocol_digest": keyed_prg_protocol_record()[
                "keyed_prg_protocol_digest"
            ],
            "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
        },
        "candidate_matrix_content_sha256": "1" * 64,
        "risk_budget_content_sha256": "2" * 64,
        "response_matrix_content_sha256": "3" * 64,
        "latent_basis_content_sha256": "4" * 64,
    }
    config = SemanticWatermarkRuntimeConfig()
    qk_atomic_content_records = []
    qk_values = torch.zeros(1, 4, 4)
    qk_probabilities = torch.softmax(qk_values, dim=-1)
    for layer_name in config.attention_module_names:
        qk_atomic_content_records.append(
            build_qk_atomic_content_metadata(
                layer_name,
                qk_values,
                qk_values,
                qk_values,
                qk_probabilities,
                (0, 1, 2, 3),
            )
        )
    qk_atomic_digest = qk_atomic_content_records_digest(
        qk_atomic_content_records
    )
    attention_qk_atomic_content_records = [
        {
            "qk_evaluation_role": role,
            "qk_atomic_content_records": qk_atomic_content_records,
            "qk_atomic_content_digest": qk_atomic_digest,
            "qk_atomic_content_ready": True,
        }
        for role in (
            "latent_before",
            "content_base_latent",
            "accepted_attention_candidate",
            "actual_written_combined_latent",
        )
    ]
    branch_update_content_records = {
        "lf_content": "5" * 64,
        "tail_robust": "6" * 64,
        "attention_geometry": "7" * 64,
    }
    record = {
        "step_index": 6,
        "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
        "adjacent_step_reference_index": 5,
        "adjacent_step_reference_latent_content_sha256": "1" * 64,
        "adjacent_step_stability_status": (
            "measured_from_immediately_previous_scheduler_step"
        ),
        "branch_risk_bundle_digest": "risk_digest",
        "branch_risk_records": {
            name: {
                "eligible_position_count": 10,
                "risk_values_content_sha256": "8" * 64,
                "budget_values_content_sha256": "9" * 64,
                "eligible_mask_content_sha256": "a" * 64,
            }
            for name in ("lf_content", "tail_robust", "attention_geometry")
        },
        "branch_risk_content_digest": build_stable_digest(
            {
                name: {
                    "risk_values_content_sha256": "8" * 64,
                    "budget_values_content_sha256": "9" * 64,
                    "eligible_mask_content_sha256": "a" * 64,
                }
                for name in (
                    "lf_content",
                    "tail_robust",
                    "attention_geometry",
                )
            }
        ),
        "null_space_records": {
            "lf_content": dict(subspace),
            "tail_robust": dict(subspace),
            "attention_geometry": dict(subspace),
        },
        "lf_projection_energy_retention": 0.2,
        "tail_projection_energy_retention": 0.2,
        "attention_score_gain": 0.01,
        "attention_applied_update_strength": 0.001,
        "stable_token_indices": [0, 1, 2, 3],
        "stable_token_selection_digest": "e" * 64,
        "stable_pair_weight_identity_digest": "f" * 64,
        "stable_pair_weight_realization_digest": "a" * 64,
        "attention_relation_component_names": list(
            ATTENTION_RELATION_COMPONENT_NAMES
        ),
        "attention_relation_active_component_names": list(
            attention_relation_component_protocol(
                config.attention_relation_component_weights
            )["attention_relation_active_component_names"]
        ),
        "attention_relation_component_weights": list(
            config.attention_relation_component_weights
        ),
        "attention_relation_component_protocol_digest": (
            attention_relation_component_protocol(
                config.attention_relation_component_weights
            )["attention_relation_component_protocol_digest"]
        ),
        "attention_relation_source": (
            "direct_qk_centered_logits_and_probabilities"
        ),
        "attention_relation_direct_qk_source_ready": True,
        "attention_relation_probability_scope": (
            "sampled_image_token_qk_relation_probability"
        ),
        "attention_relation_component_identity_digest": "b" * 64,
        "attention_relation_keyed_projection_digest": "c" * 64,
        "attention_qk_atomic_content_records": (
            attention_qk_atomic_content_records
        ),
        "attention_qk_atomic_content_digest": build_stable_digest(
            {
                "attention_qk_atomic_content_records": (
                    attention_qk_atomic_content_records
                )
            }
        ),
        "attention_qk_atomic_content_ready": True,
        "attention_relation_qk_operator_metadata_records": [
            {
                "record_layer_name": layer_name,
                "coordinate_convention": ATTENTION_COORDINATE_CONVENTION,
                "grid_align_corners": ATTENTION_GRID_ALIGN_CORNERS,
            }
            for layer_name in config.attention_module_names
        ],
        "attention_relation_qk_operator_metadata_ready": True,
        "attention_module_names": list(config.attention_module_names),
        "attention_coordinate_convention": (
            config.attention_coordinate_convention
        ),
        "attention_grid_align_corners": (
            config.attention_grid_align_corners
        ),
        "quantized_write_update_content_sha256": "d" * 64,
        "lf_update_content_sha256": (
            branch_update_content_records["lf_content"]
        ),
        "tail_robust_update_content_sha256": (
            branch_update_content_records["tail_robust"]
        ),
        "attention_geometry_update_content_sha256": (
            branch_update_content_records["attention_geometry"]
        ),
        "branch_updates_content_digest": build_stable_digest(
            branch_update_content_records
        ),
        "quantized_write_jacobian_gate_applicable": True,
        "quantized_write_jacobian_response_norm": 1e-5,
        "quantized_write_reference_feature_norm": 1.0,
        "quantized_write_relative_jacobian_response": 1e-5,
        "maximum_quantized_write_relative_jacobian_response": 1e-4,
        "quantized_write_jacobian_gate_ready": True,
        "quantized_write_jacobian_status": (
            "measured_from_actual_quantized_latent_delta"
        ),
        "keyed_prg_version": config.keyed_prg_version,
        "keyed_prg_protocol_digest": keyed_prg_protocol_record(
            config.keyed_prg_version
        )["keyed_prg_protocol_digest"],
        "full_semantic_cosine_similarity": 0.999,
        "full_handcrafted_structure_feature_relative_drift": 0.001,
        "semantic_preservation_gate_ready": True,
    }
    assert _scientific_update_record_ready(record, config) is True
    record["attention_score_gain"] = 0.0
    assert _scientific_update_record_ready(record, config) is False
    record["attention_score_gain"] = 0.01
    record["full_semantic_cosine_similarity"] = 0.9
    assert _scientific_update_record_ready(record, config) is False
    record["full_semantic_cosine_similarity"] = 0.999
    record["quantized_write_relative_jacobian_response"] = 1e-3
    assert _scientific_update_record_ready(record, config) is False
    record["quantized_write_relative_jacobian_response"] = 1e-5
    record["keyed_prg_protocol_digest"] = "0" * 64
    assert _scientific_update_record_ready(record, config) is False
    record["keyed_prg_protocol_digest"] = keyed_prg_protocol_record()[
        "keyed_prg_protocol_digest"
    ]
    record["adjacent_step_reference_index"] = 4
    assert _scientific_update_record_ready(record, config) is False
    record["adjacent_step_reference_index"] = 5
    record["attention_module_names"] = ["transformer_blocks.1.attn"]
    assert _scientific_update_record_ready(record, config) is False


@pytest.mark.quick
def test_tail_robust_template_records_amplitude_tail_semantics() -> None:
    """尾部截断应改变稀疏率, 并记录幅值尾部语义。"""

    latent = torch.zeros(1, 2, 8, 8)
    lf_template = build_low_frequency_template(latent, "key", "model")
    tail_template, _, retained_fraction = build_tail_robust_template(latent, "key", "model", 0.20)
    observed = 0.7 * lf_template + 0.3 * tail_template
    score = compute_blind_content_score(observed, lf_template, tail_template)

    assert 0.15 <= retained_fraction <= 0.25
    assert score.content_score > 0.5
    assert score.metadata["tail_branch_semantics"] == "gaussian_amplitude_tail_truncation"


@pytest.mark.quick
def test_keyed_templates_use_versioned_device_independent_prg() -> None:
    """密钥模板必须由固定 PRG 算法生成, 设备 RNG 不得参与定义."""

    reference = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    first_lf = build_low_frequency_template(
        reference,
        "known-key",
        "known-model",
    )
    second_lf = build_low_frequency_template(
        reference,
        "known-key",
        "known-model",
    )
    tail, threshold, retained_fraction = build_tail_robust_template(
        reference,
        "known-key",
        "known-model",
        0.25,
    )
    protocol = keyed_prg_protocol_record()

    assert KEYED_PRG_VERSION == (
        "sha256_counter_box_muller_float32_v1"
    )
    assert torch.equal(first_lf, second_lf)
    assert hashlib.sha256(
        first_lf.detach().contiguous().numpy().tobytes()
    ).hexdigest() == "7bb7ce000ed31ce3470d4554424849c7602d735de73a9bed4e3bed491ba65f8e"
    assert hashlib.sha256(
        tail.detach().contiguous().numpy().tobytes()
    ).hexdigest() == "b213af9ced637ddf82540806b18a57b5ecc1e94d71e2256f726ce3980612f7a8"
    assert threshold == pytest.approx(1.4844163656234741)
    assert retained_fraction == 0.25
    assert protocol["canonical_generation_device"] == "cpu"
    assert len(protocol["keyed_prg_protocol_digest"]) == 64

    first_candidates = generate_keyed_candidate_directions(
        torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        "known-key",
        "lf_content",
        2,
    )
    second_candidates = generate_keyed_candidate_directions(
        torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        "known-key",
        "lf_content",
        2,
    )
    relation_signs = keyed_relation_signs(
        torch.zeros((4, 4), dtype=torch.float32),
        "known-key",
        "transformer_blocks.0.attn",
    )
    assert torch.equal(first_candidates, second_candidates)
    assert torch.equal(relation_signs, relation_signs.transpose(0, 1))
    assert torch.count_nonzero(torch.diag(relation_signs)).item() == 0

    source = "\n".join(
        inspect.getsource(module)
        for module in (
            keyed_prg_module,
            keyed_tensor_module,
            attention_module,
            nullspace_module,
        )
    )
    assert "torch.Generator(" not in source
    assert "torch.randn(" not in source
    assert "torch.quantile(" not in source

    with pytest.raises(ValueError, match="keyed_prg_version"):
        build_low_frequency_template(
            reference,
            "known-key",
            "known-model",
            prg_version="unsupported_prg",
        )
    with pytest.raises(ValueError, match="keyed_prg_version"):
        generate_keyed_candidate_directions(
            torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            "known-key",
            "lf_content",
            2,
            prg_version="unsupported_prg",
        )


class _ToyAttention(torch.nn.Module):
    """提供真实 Q/K 投影的轻量注意力模块。"""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(width, width, bias=False)
        self.to_k = torch.nn.Linear(width, width, bias=False)
        self.heads = 1
        torch.nn.init.eye_(self.to_q.weight)
        torch.nn.init.eye_(self.to_k.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """返回输入, Q/K 由正式记录钩子直接读取。"""

        return hidden_states


@pytest.mark.quick
def test_attention_modules_resolve_only_frozen_exact_layer_names() -> None:
    """运行时必须按配置层名解析模块, 不得按枚举位置重新选择."""

    class ToyBlock(torch.nn.Module):
        """提供一个带主注意力模块的轻量 Transformer block."""

        def __init__(self) -> None:
            super().__init__()
            self.attn = _ToyAttention(4)

    class ToyTransformer(torch.nn.Module):
        """提供与 SD3.5 相同的公开 block 路径结构."""

        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(
                ToyBlock() for _ in range(24)
            )

    transformer = ToyTransformer()
    pipeline = SimpleNamespace(transformer=transformer)
    frozen_names = (
        "transformer_blocks.0.attn",
        "transformer_blocks.23.attn",
    )

    resolved = _attention_modules(pipeline, frozen_names)

    assert tuple(name for name, _ in resolved) == frozen_names
    assert resolved[0][1] is transformer.transformer_blocks[0].attn
    assert resolved[1][1] is transformer.transformer_blocks[23].attn
    with pytest.raises(RuntimeError, match="冻结注意力层不存在"):
        _attention_modules(
            pipeline,
            ("transformer_blocks.0.attn", "transformer_blocks.24.attn"),
        )


@pytest.mark.quick
def test_image_alignment_uses_token_endpoint_coordinate_convention() -> None:
    """图像重采样必须使用与 token 角点中心一致的 align_corners=True."""

    image = Image.new("RGB", (3, 3))
    image.putdata(
        [
            (value, value, value)
            for _ in range(3)
            for value in (10, 100, 200)
        ]
    )
    alignment = SimpleNamespace(
        affine_transform=((1.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    )

    aligned = _align_image(image, alignment)

    assert ATTENTION_COORDINATE_CONVENTION == (
        "normalized_xy_token_centers_corner_endpoints_v1"
    )
    assert ATTENTION_GRID_ALIGN_CORNERS is True
    assert [aligned.getpixel((column, 1))[0] for column in range(3)] == [
        100,
        200,
        200,
    ]


@pytest.mark.quick
def test_qk_sampling_preserves_two_dimensional_token_grid() -> None:
    """有界 Q/K 抽样必须沿二维行列轴取点, 不能等距抽一维序号。"""

    module = _ToyAttention(4)
    hidden_states = torch.randn(1, 16, 4)

    relation, indices = qk_self_attention(module, hidden_states, max_tokens=4)

    assert indices == (0, 3, 12, 15)
    assert isinstance(relation, QKAttentionRelation)
    assert relation.relation_source == (
        "direct_qk_centered_logits_and_probabilities"
    )
    assert relation.centered_logits.shape == relation.probabilities.shape


@pytest.mark.quick
def test_multihead_qk_relation_matches_independent_manual_calculation() -> None:
    """多头 logits 与概率必须先逐头计算再分别平均, 且记录可读算子元数据。"""

    module = _ToyAttention(4)
    module.heads = 2
    module.scale = 1.0 / (2.0**0.5)
    hidden_states = torch.tensor(
        [
            [
                [0.1, 0.8, -0.4, 0.2],
                [0.7, -0.2, 0.3, 0.9],
                [-0.5, 0.4, 0.6, -0.1],
                [0.2, -0.7, 0.5, 0.3],
            ]
        ]
    )
    relation, token_indices = qk_self_attention(
        module,
        hidden_states,
        max_tokens=4,
        layer_name="manual_multihead_layer",
    )
    per_head = hidden_states.reshape(1, 4, 2, 2).transpose(1, 2)
    logits = torch.matmul(per_head, per_head.transpose(-1, -2)) * module.scale
    expected_centered = (
        logits - logits.mean(dim=-1, keepdim=True)
    ).mean(dim=1)
    expected_probability = torch.softmax(logits, dim=-1).mean(dim=1)
    softmax_of_mean_logits = torch.softmax(logits.mean(dim=1), dim=-1)

    assert token_indices == (0, 1, 2, 3)
    assert torch.allclose(relation.centered_logits, expected_centered)
    assert torch.allclose(relation.probabilities, expected_probability)
    assert not torch.allclose(relation.probabilities, softmax_of_mean_logits)
    assert relation.metadata["module_layer_name"] == "manual_multihead_layer"
    assert relation.metadata["head_count"] == 2
    assert relation.metadata["head_width"] == 2
    assert relation.metadata["attention_scale"] == pytest.approx(module.scale)
    assert relation.metadata["q_normalization_applied"] is False
    assert relation.metadata["k_normalization_applied"] is False
    assert relation.metadata["sampled_token_indices"] == [0, 1, 2, 3]
    assert relation.metadata["coordinate_convention"] == (
        ATTENTION_COORDINATE_CONVENTION
    )
    assert relation.metadata["grid_align_corners"] is True
    assert relation.metadata[
        "mean_probability_is_softmax_of_mean_logits"
    ] is False
    identity = build_attention_relation_graph_identity(
        (("manual_multihead_layer", relation, token_indices),),
        "manual_multihead_key",
    )
    assert identity.qk_operator_metadata_ready is True
    assert len(identity.qk_operator_metadata_digest) == 64
    assert identity.qk_atomic_content_ready is True
    assert len(identity.qk_atomic_content_records) == 1
    assert len(identity.qk_atomic_content_digest) == 64


@pytest.mark.quick
def test_qk_relation_rejects_module_scale_mismatch() -> None:
    """模块公开 scale 与 head width 理论尺度不一致时必须立即失败。"""

    module = _ToyAttention(4)
    module.heads = 2
    module.scale = 0.5

    with pytest.raises(RuntimeError, match="scale"):
        qk_self_attention(
            module,
            torch.randn(1, 4, 4),
            max_tokens=4,
            layer_name="mismatched_scale_layer",
        )


@pytest.mark.quick
def test_attention_stability_comes_from_multiple_real_qk_layers() -> None:
    """相同 Q/K 关系层应产生接近 1 的真实关系稳定图。"""

    attention = torch.softmax(torch.randn(1, 4, 4), dim=-1)
    records = (
        ("layer_a", attention, (0, 3, 12, 15)),
        ("layer_b", attention.clone(), (0, 3, 12, 15)),
    )

    stability = attention_relation_stability_map(records, (4, 4))

    assert stability.shape == (1, 4, 4)
    assert float(stability.min()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.quick
def test_stable_attention_tokens_drive_keyed_geometry_score() -> None:
    """稳定 token 集必须真实改变 Q/K 目标权重并保存可复现身份。"""

    generator = torch.Generator().manual_seed(1703)
    base = torch.softmax(torch.randn(1, 9, 9, generator=generator), dim=-1)
    token_indices = tuple(range(9))
    records = (
        ("layer_a", base, token_indices),
        ("layer_b", base.clone(), token_indices),
    )

    selection = select_stable_attention_tokens(records, stable_token_fraction=0.5)
    pair_weights = build_stable_attention_pair_weights(
        records,
        selection,
        unstable_pair_weight=0.0,
    )
    weighted = attention_geometry_score(
        records,
        "stable_token_key",
        stable_pair_weights=pair_weights,
    )
    full = attention_geometry_score(
        records,
        "stable_token_key",
        stable_token_positions=selection.token_positions,
        unstable_pair_weight=0.99,
    )

    assert len(selection.token_indices) == 5
    assert len(selection.selection_digest) == 64
    assert len(pair_weights.pair_weight_identity_digest) == 64
    assert float(weighted) != pytest.approx(float(full), abs=1e-8)


@pytest.mark.quick
def test_each_attention_relation_component_changes_keyed_score() -> None:
    """四个非冗余分量逐一变化时, 密钥分量投影总分都必须发生变化。"""

    generator = torch.Generator().manual_seed(260712)
    logits = torch.randn(1, 9, 9, generator=generator)
    relation = _direct_qk_relation_from_logits(logits)
    token_indices = tuple(range(9))
    descriptor = build_attention_relation_descriptor(relation, token_indices)
    projection = keyed_attention_relation_projection(
        descriptor,
        "four_component_key",
        "four_component_layer",
    )
    pair_weights = 1.0 - torch.eye(9)
    baseline_components = attention_relation_component_scores(
        descriptor.values,
        projection.values,
        pair_weights,
    )
    baseline_total = baseline_components.mean(dim=-1)

    for component_index in range(len(ATTENTION_RELATION_COMPONENT_NAMES)):
        changed_values = descriptor.values.clone()
        changed_values[..., component_index] = (
            changed_values[..., component_index]
            + 0.5 * projection.values[..., component_index]
        )
        changed_components = attention_relation_component_scores(
            changed_values,
            projection.values,
            pair_weights,
        )
        changed_total = changed_components.mean(dim=-1)
        assert not torch.allclose(
            changed_components[..., component_index],
            baseline_components[..., component_index],
            atol=1e-6,
            rtol=0.0,
        )
        assert not torch.allclose(
            changed_total,
            baseline_total,
            atol=1e-6,
            rtol=0.0,
        )


@pytest.mark.quick
def test_leave_one_component_out_weights_remove_exact_score_contribution() -> None:
    """留一权重协议必须让被移除分量不再进入真实组合分数."""

    component_scores = torch.tensor((0.2, -0.4, 0.6, 0.8))
    for removed_index in range(len(ATTENTION_RELATION_COMPONENT_NAMES)):
        weights = tuple(
            0.0 if index == removed_index else 1.0 / 3.0
            for index in range(len(ATTENTION_RELATION_COMPONENT_NAMES))
        )
        baseline = combine_attention_relation_component_scores(
            component_scores,
            weights,
        )
        changed = component_scores.clone()
        changed[removed_index] += 100.0
        changed_score = combine_attention_relation_component_scores(
            changed,
            weights,
        )
        protocol = attention_relation_component_protocol(weights)

        assert float(changed_score) == pytest.approx(float(baseline))
        assert ATTENTION_RELATION_COMPONENT_NAMES[removed_index] not in (
            protocol["attention_relation_active_component_names"]
        )
        assert len(protocol["attention_relation_component_protocol_digest"]) == 64


@pytest.mark.quick
def test_differentiable_soft_rank_contributes_nonzero_logit_gradient() -> None:
    """soft-rank 分量必须对真实 Q/K logits 保留非零可微梯度。"""

    generator = torch.Generator().manual_seed(260713)
    logits = torch.randn(1, 9, 9, generator=generator).requires_grad_(True)
    relation = QKAttentionRelation(
        centered_logits=logits - logits.mean(dim=-1, keepdim=True),
        probabilities=torch.softmax(logits, dim=-1),
    )
    descriptor = build_attention_relation_descriptor(relation, tuple(range(9)))
    projection = keyed_attention_relation_projection(
        descriptor,
        "soft_rank_gradient_key",
        "soft_rank_gradient_layer",
    )
    component_scores = attention_relation_component_scores(
        descriptor.values,
        projection.values,
        1.0 - torch.eye(9),
    )
    gradient = torch.autograd.grad(component_scores[..., 1].sum(), logits)[0]

    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.norm()) > 1e-6


@pytest.mark.quick
def test_distance_modulated_probability_is_distinct_and_differentiable() -> None:
    """距离调制概率必须区别于 P, 并对真实 Q/K logits 保留非零梯度。"""

    generator = torch.Generator().manual_seed(260714)
    logits = torch.randn(1, 9, 9, generator=generator).requires_grad_(True)
    relation = QKAttentionRelation(
        centered_logits=logits - logits.mean(dim=-1, keepdim=True),
        probabilities=torch.softmax(logits, dim=-1),
    )
    descriptor = build_attention_relation_descriptor(relation, tuple(range(9)))
    projection = keyed_attention_relation_projection(
        descriptor,
        "distance_modulation_key",
        "distance_modulation_layer",
    )
    component_scores = attention_relation_component_scores(
        descriptor.values,
        projection.values,
        1.0 - torch.eye(9),
    )
    gradient = torch.autograd.grad(component_scores[..., 3].sum(), logits)[0]

    assert not torch.allclose(
        descriptor.values[..., 2],
        descriptor.values[..., 3],
    )
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.norm()) > 1e-6


def _identity_null_space(latent: torch.Tensor) -> JacobianNullSpaceResult:
    """构造完整空间基底, 隔离注意力梯度测试。"""

    element_count = latent.numel()
    identity = torch.eye(element_count)
    response_matrix = torch.zeros(1, element_count)
    risk_budget = torch.ones_like(latent, dtype=torch.float32)
    return JacobianNullSpaceResult(
        branch_name="attention_geometry",
        candidate_matrix=identity,
        response_matrix=response_matrix,
        latent_basis=identity,
        column_response_norms=(0.0,) * element_count,
        column_relative_response_residuals=(0.0,) * element_count,
        projection_energy_retentions=(1.0,) * element_count,
        cg_iteration_counts=(0,) * element_count,
        cg_relative_residuals=(0.0,) * element_count,
        evaluated_direction_indices=tuple(range(element_count)),
        response_residual=0.0,
        relative_response_residual=0.0,
        orthogonality_error=0.0,
        candidate_matrix_content_sha256=tensor_content_sha256(identity),
        risk_budget_content_sha256=tensor_content_sha256(risk_budget),
        response_matrix_content_sha256=tensor_content_sha256(response_matrix),
        latent_basis_content_sha256=tensor_content_sha256(identity),
        solver_digest="identity_test_basis",
        metadata={},
    )


@pytest.mark.quick
def test_attention_update_uses_real_qk_and_autograd() -> None:
    """注意力几何更新必须来自真实 Q/K 投影和 latent autograd。"""

    module = _ToyAttention(4)
    latent = torch.tensor(
        [[[0.3, 0.2, -0.1, 0.4], [0.1, -0.2, 0.5, 0.3], [-0.4, 0.2, 0.1, 0.6], [0.2, 0.7, -0.3, 0.1]]]
    )
    attention, indices = qk_self_attention(module, latent, max_tokens=4)
    assert attention.shape == (1, 4, 4)
    assert indices == (0, 1, 2, 3)

    with DifferentiableAttentionRecorder(
        (("toy_attention_a", module), ("toy_attention_b", module)),
        max_tokens=4,
    ) as recorder:
        update = optimize_attention_geometry_update(
            latent=latent,
            transformer_forward=module,
            recorder=recorder,
            key_material="attention_key",
            safe_subspace=_identity_null_space(latent),
            update_strength=0.05,
        )

    assert update.gradient_norm > 0.0
    assert update.projected_gradient_norm > 0.0
    assert update.score_after >= update.score_before - 1e-6
    assert update.metadata["attention_source"] == "real_qk_projection"
    assert update.metadata["gradient_source"] == "torch_autograd"
    assert update.qk_atomic_content_ready is True
    assert tuple(
        record["qk_evaluation_role"]
        for record in update.qk_atomic_evaluation_records
    ) == (
        "latent_before",
        "content_base_latent",
        "accepted_attention_candidate",
    )
    assert len(update.qk_atomic_evaluation_digest) == 64


@pytest.mark.quick
def test_attention_update_verifies_actual_combined_latent() -> None:
    """Attention 回溯必须以固定内容更新为基底并验证真正写回的组合 latent。"""

    module = _ToyAttention(4)
    latent = torch.tensor(
        [[[0.3, 0.2, -0.1, 0.4], [0.1, -0.2, 0.5, 0.3], [-0.4, 0.2, 0.1, 0.6], [0.2, 0.7, -0.3, 0.1]]]
    )
    content_base_update = torch.tensor(
        [[[0.01, -0.02, 0.01, 0.00], [0.00, 0.01, -0.01, 0.02], [0.01, 0.00, 0.02, -0.01], [-0.02, 0.01, 0.00, 0.01]]]
    )
    with DifferentiableAttentionRecorder(
        (("toy_attention_a", module), ("toy_attention_b", module)),
        max_tokens=4,
    ) as recorder:
        update = optimize_attention_geometry_update(
            latent=latent,
            transformer_forward=module,
            recorder=recorder,
            key_material="combined_attention_key",
            safe_subspace=_identity_null_space(latent),
            update_strength=0.05,
            base_update=content_base_update,
        )
        recorder.clear()
        module(latent + content_base_update + update.update)
        actual_score = float(
            attention_geometry_score(
                recorder.records,
                "combined_attention_key",
            ).detach().item()
        )

    assert actual_score == pytest.approx(update.score_after, abs=1e-7)
    assert actual_score > update.score_before
    assert actual_score > update.content_base_score
    assert update.metadata["verified_candidate"] == "actual_combined_latent"


@pytest.mark.quick
@pytest.mark.parametrize(
    ("transform_name", "permutation"),
    (
        ("horizontal_flip", tuple(row * 8 + (7 - column) for row in range(8) for column in range(8))),
        ("vertical_flip", tuple((7 - row) * 8 + column for row in range(8) for column in range(8))),
        ("rotation_90", tuple((7 - column) * 8 + row for row in range(8) for column in range(8))),
    ),
)
def test_attention_registration_is_equivariant_to_query_and_key_permutation(
    transform_name: str,
    permutation: tuple[int, ...],
) -> None:
    """注册必须同时还原 ``P A P^T`` 的查询轴和键轴。"""

    token_count = 64
    key_material = "equivariant_registration_key"
    layer_name = "registered_layer"
    relation_signs = keyed_relation_signs(
        torch.zeros(1, token_count, token_count),
        key_material,
        layer_name,
    )
    canonical_logits = (2.0 * relation_signs).unsqueeze(0)
    index = torch.tensor(permutation, dtype=torch.long)
    observed = _direct_qk_relation_from_logits(
        canonical_logits.index_select(1, index).index_select(2, index),
        layer_name,
    )

    result = recover_attention_affine_alignment(
        observed,
        key_material,
        layer_name,
        tuple(range(token_count)),
        build_stable_attention_pair_weights(
            (
                (layer_name, observed, tuple(range(token_count))),
                (f"{layer_name}_replicate", observed.clone(), tuple(range(token_count))),
            ),
            select_stable_attention_tokens(
                (
                    (layer_name, observed, tuple(range(token_count))),
                    (f"{layer_name}_replicate", observed.clone(), tuple(range(token_count))),
                )
            ),
        ),
    )

    assert transform_name
    assert result.geometry_reliable is True
    assert result.inlier_ratio == pytest.approx(1.0)
    assert result.relation_sync_score > 0.65
    assert set(result.relation_component_scores) == set(
        ATTENTION_RELATION_COMPONENT_NAMES
    )
    assert result.metadata["attention_relation_direct_qk_source_ready"] is True
    assert result.metadata["matcher"] == "double_sided_keyed_relation_graph_registration"
    assert result.metadata["stable_pair_weight_identity_ready"] is True


@pytest.mark.quick
def test_image_only_detector_reextracts_qk_after_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """几何可靠性必须包含图像对齐后重新提取的真实 Q/K sync。"""

    token_count = 64
    key_material = "detector_sync_key"
    model_id = "detector_sync_model"
    layer_name = "detector_sync_layer"
    relation_signs = keyed_relation_signs(
        torch.zeros(1, token_count, token_count),
        key_material,
        layer_name,
    )
    canonical_logits = (2.0 * relation_signs).unsqueeze(0)
    canonical_attention = _direct_qk_relation_from_logits(
        canonical_logits,
        layer_name,
    )
    flip = torch.tensor(
        [row * 8 + (7 - column) for row in range(8) for column in range(8)],
        dtype=torch.long,
    )
    observed_attention = _direct_qk_relation_from_logits(
        canonical_logits.index_select(1, flip).index_select(2, flip),
        layer_name,
    )
    reference = torch.zeros(1, 2, 8, 8)
    lf_template = build_low_frequency_template(reference, key_material, model_id)
    tail_template = build_tail_robust_template(reference, key_material, model_id, 0.20)[0]
    original = {"latent": torch.zeros_like(reference), "attention": observed_attention}
    aligned = {
        "latent": 0.8 * lf_template + 0.4 * tail_template,
        "attention": canonical_attention,
    }
    extraction_count = 0
    stable_selection_count = 0

    import main.methods.detection.image_only as detector_module

    original_select_stable_tokens = detector_module.select_stable_attention_tokens

    def select_stable_tokens_once(*args: object, **kwargs: object):
        """记录盲检稳定 token 选择次数, 对齐后不得再次选择。"""

        nonlocal stable_selection_count
        stable_selection_count += 1
        return original_select_stable_tokens(*args, **kwargs)

    monkeypatch.setattr(
        detector_module,
        "select_stable_attention_tokens",
        select_stable_tokens_once,
    )

    def extract(sample: dict[str, object]) -> tuple[tuple[str, object, tuple[int, ...]], ...]:
        nonlocal extraction_count
        extraction_count += 1
        record = (layer_name, sample["attention"], tuple(range(token_count)))
        return (record, record)

    result = detect_image_only_watermark(
        image=original,
        key_material=key_material,
        config=ImageOnlyDetectionConfig(
            model_id=model_id,
            content_threshold=0.2,
            geometry_score_threshold=0.5,
            registration_confidence_threshold=0.5,
            attention_sync_score_threshold=0.5,
            rescue_margin_low=-0.5,
            attention_relation_component_weights=(
                1.0 / 3.0,
                0.0,
                1.0 / 3.0,
                1.0 / 3.0,
            ),
        ),
        image_latent_encoder=lambda sample: sample["latent"],
        image_attention_extractor=extract,
        image_aligner=lambda _image, _alignment: aligned,
    )

    assert extraction_count == 2
    assert stable_selection_count == 1
    assert result.raw_attention_geometry_score is not None
    assert result.attention_geometry_score is not None
    assert result.attention_geometry_score > 0.65
    assert result.attention_sync_score is not None and result.attention_sync_score > 0.65
    assert result.metadata["attention_relation_direct_qk_source_ready"] is True
    assert result.geometry_reliable is True
    assert result.rescue_applied is True
    assert result.metadata["stable_pair_weight_identity_ready"] is True
    assert len(result.metadata["stable_pair_weight_identity_digest"]) == 64
    assert len(result.metadata["attention_record_schema_digest"]) == 64
    assert result.metadata["detection_qk_atomic_content_ready"] is True
    assert tuple(
        record["qk_evaluation_role"]
        for record in result.metadata["detection_qk_atomic_content_records"]
    ) == ("raw_detection_image", "aligned_detection_image")
    assert len(result.metadata["detection_qk_atomic_content_digest"]) == 64
    assert result.metadata["attention_relation_component_weights"] == [
        1.0 / 3.0,
        0.0,
        1.0 / 3.0,
        1.0 / 3.0,
    ]
    assert "differentiable_row_rank" not in result.metadata[
        "attention_relation_active_component_names"
    ]
    assert len(
        result.metadata["attention_relation_component_protocol_digest"]
    ) == 64
    assert result.alignment is not None
    assert result.alignment.attention_relation_component_weights == (
        1.0 / 3.0,
        0.0,
        1.0 / 3.0,
        1.0 / 3.0,
    )


@pytest.mark.quick
def test_image_attention_extractor_batches_flowmatch_timestep(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlowMatch ``scale_noise`` 必须接收与 latent batch 一致的一维 timestep。"""

    import experiments.runners.semantic_watermark_runtime as runtime_module

    module = _ToyAttention(1)

    class Scheduler:
        """记录仅图像检测传入的 timestep 形状。"""

        def __init__(self) -> None:
            self.timesteps = torch.arange(20, dtype=torch.float32)
            self.received_timestep: torch.Tensor | None = None
            self.schedule_step_counts: list[int] = []

        def set_timesteps(self, step_count: int, device: str) -> None:
            self.schedule_step_counts.append(step_count)
            self.timesteps = torch.arange(step_count, device=device, dtype=torch.float32)

        def scale_noise(
            self,
            latent: torch.Tensor,
            timestep: torch.Tensor,
            noise: torch.Tensor,
        ) -> torch.Tensor:
            self.received_timestep = timestep
            assert timestep.shape == (latent.shape[0],)
            return latent + 0.0 * noise

    scheduler = Scheduler()
    pipeline = SimpleNamespace(scheduler=scheduler, _execution_device="cpu")
    monkeypatch.setattr(
        runtime_module,
        "_encode_image_latent",
        lambda _pipeline, _image: torch.zeros(2, 1, 2, 2),
    )
    monkeypatch.setattr(
        runtime_module,
        "_transformer_forward_function",
        lambda *_args, **_kwargs: lambda latent: module(latent.reshape(latent.shape[0], 4, 1)),
    )
    config = SemanticWatermarkRuntimeConfig()
    extractor = _image_attention_extractor(
        pipeline,
        config,
        (("toy_attention", module),),
        None,
        None,
    )

    records = extractor(object())

    assert records
    assert scheduler.received_timestep is not None
    assert scheduler.received_timestep.shape == (2,)
    assert scheduler.received_timestep[0].item() == pytest.approx(7.0)
    assert scheduler.schedule_step_counts == [20]

    # 模拟共享 img2img scheduler 在扩散攻击结束后留下另一套日程. 下一次盲检
    # 必须重新建立20步正式检测日程, 不能把旧 timestep 与攻击 sigma 混用.
    scheduler.set_timesteps(5, "cpu")
    scheduler.received_timestep = None
    second_records = extractor(object())

    assert second_records
    assert scheduler.schedule_step_counts == [20, 5, 20]
    assert scheduler.received_timestep is not None
    assert scheduler.received_timestep.shape == (2,)
    assert scheduler.received_timestep[0].item() == pytest.approx(7.0)


@pytest.mark.quick
def test_image_attention_extractor_requires_scheduler_scale_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式 Q/K 提取必须使用 scheduler 的真实加噪算子, 不得线性替代."""

    import experiments.runners.semantic_watermark_runtime as runtime_module

    class SchedulerWithoutScaleNoise:
        """只提供检测日程, 故意缺少 scale_noise."""

        def __init__(self) -> None:
            self.timesteps = torch.arange(20, dtype=torch.float32)

        def set_timesteps(self, step_count: int, device: str) -> None:
            self.timesteps = torch.arange(
                step_count,
                device=device,
                dtype=torch.float32,
            )

    pipeline = SimpleNamespace(
        scheduler=SchedulerWithoutScaleNoise(),
        _execution_device="cpu",
    )
    monkeypatch.setattr(
        runtime_module,
        "_encode_image_latent",
        lambda _pipeline, _image: torch.zeros(1, 1, 2, 2),
    )
    extractor = _image_attention_extractor(
        pipeline,
        SemanticWatermarkRuntimeConfig(),
        (),
        None,
        None,
    )

    with pytest.raises(RuntimeError, match="scheduler.*scale_noise"):
        extractor(object())


@pytest.mark.quick
def test_post_step_injection_requires_adjacent_scheduler_steps() -> None:
    """注入时刻必须同时具有真实的上一和下一 scheduler 时刻."""

    base = SemanticWatermarkRuntimeConfig()
    with pytest.raises(ValueError, match="post-step"):
        replace(base, injection_step_indices=(base.inference_steps - 1,))
    with pytest.raises(ValueError, match="相邻的前后调度时刻"):
        replace(base, injection_step_indices=(0,))


@pytest.mark.quick
def test_image_only_detector_interface_and_positive_content_path() -> None:
    """正式检测接口不得接收生成轨迹, 且能从图像编码 latent 完成内容主判。"""

    parameters = set(inspect.signature(detect_image_only_watermark).parameters)
    assert "generation_latent_trace" not in parameters
    assert "source_latent" not in parameters
    assert "prompt" not in parameters

    reference = torch.zeros(1, 2, 8, 8)
    lf_template = build_low_frequency_template(reference, "blind_key", "model")
    tail_template = build_tail_robust_template(reference, "blind_key", "model", 0.20)[0]
    encoded = 0.8 * lf_template + 0.4 * tail_template
    result = detect_image_only_watermark(
        image=encoded,
        key_material="blind_key",
        config=ImageOnlyDetectionConfig(
            model_id="model",
            content_threshold=0.20,
            geometry_score_threshold=0.0,
        ),
        image_latent_encoder=lambda image: image,
    )

    assert result.positive_by_content is True
    assert result.evidence_positive is True
    assert result.content_failure_reason == "content_positive"
    assert result.rescue_applied is False
    assert result.metadata["blind_image_detector"] is True
    assert result.metadata["generation_latent_trace_required"] is False


@pytest.mark.quick
def test_complete_evidence_calibration_includes_geometry_rescue() -> None:
    """阈值搜索必须直接约束加入同阈值 rescue 后的完整误报率。"""

    calibration_records = []
    for index in range(33):
        calibration_records.append(
            {
                "content_score": index / 100.0,
                "aligned_content_score": (index + 5) / 100.0,
                "geometry_reliable": index % 2 == 0,
                "attention_geometry_score": 0.5 + index / 1000.0,
                "registration_confidence": 0.6 + index / 1000.0,
                "attention_sync_score": 0.7 + index / 1000.0,
                "alignment": {
                    "registration_geometry_reliable": index % 2 == 0,
                    "geometry_reliable": index % 2 == 0,
                },
            }
        )
    protocol = calibrate_complete_evidence_protocol(
        calibration_records,
        target_fpr=0.1,
        rescue_margin_low=-0.05,
    )
    formal_records = apply_frozen_evidence_protocol(calibration_records, protocol)

    assert sum(record["formal_evidence_positive"] for record in formal_records) <= 2
    assert protocol.calibration_false_positive_count <= 2
    assert protocol.calibration_false_positive_rate <= 0.1
    assert protocol.geometry_protocol_calibration_ready is True
    assert protocol.geometry_calibration_negative_count == 33
    assert protocol.registration_calibration_negative_count == 33
    assert protocol.sync_calibration_negative_count == 33


@pytest.mark.quick
def test_geometry_protocol_cannot_close_with_missing_calibration_scores() -> None:
    """任一几何数值门禁缺失时不得把完整 rescue 协议标记为已校准。"""

    records = tuple(
        {
            "content_score": index / 100.0,
            "aligned_content_score": (index + 1) / 100.0,
            "attention_geometry_score": 0.1,
            "registration_confidence": 0.2,
            "attention_sync_score": None if index == 0 else 0.3,
            "alignment": {"registration_geometry_reliable": True},
        }
        for index in range(33)
    )

    protocol = calibrate_complete_evidence_protocol(
        records,
        target_fpr=0.1,
        rescue_margin_low=-0.05,
    )

    assert protocol.geometry_protocol_calibration_ready is False
    assert protocol.sync_calibration_negative_count == 32


@pytest.mark.quick
def test_frozen_protocol_recomputes_threshold_dependent_failure_reason() -> None:
    """冻结阈值改变后必须重算失败原因, 不能沿用预检测阈值的分类。"""

    protocol = FrozenEvidenceProtocol(
        content_threshold=0.5,
        rescue_margin_low=-0.2,
        geometry_score_threshold=0.0,
        registration_confidence_threshold=0.0,
        attention_sync_score_threshold=0.0,
        geometry_calibration_negative_count=10,
        geometry_calibration_exceedance_count=0,
        registration_calibration_negative_count=10,
        registration_calibration_exceedance_count=0,
        sync_calibration_negative_count=10,
        sync_calibration_exceedance_count=0,
        geometry_protocol_calibration_ready=True,
        calibration_negative_count=10,
        calibration_false_positive_count=0,
        calibration_false_positive_rate=0.0,
        target_fpr=0.1,
        threshold_digest="fixture_threshold",
    )
    record = {
        "content_score": 0.4,
        "aligned_content_score": 0.6,
        "attention_geometry_score": 0.1,
        "registration_confidence": 0.8,
        "attention_sync_score": 0.8,
        "geometry_reliable": False,
        "alignment": {
            "registration_geometry_reliable": True,
            "geometry_reliable": False,
        },
        "content_failure_reason": "content_positive",
    }

    resolved = apply_frozen_evidence_protocol((record,), protocol)[0]

    assert resolved["formal_content_failure_reason"] == "geometry_suspected"
    assert resolved["formal_positive_by_content"] is False
    assert resolved["formal_rescue_applied"] is True
    assert resolved["formal_evidence_positive"] is True


@pytest.mark.quick
def test_completed_runtime_cache_requires_matching_config_and_files(tmp_path: Path) -> None:
    """Colab 续跑只能复用同版本、同配置且输出完整的单 Prompt 结果。"""

    config = SemanticWatermarkRuntimeConfig(
        output_dir="outputs/cache_test",
        attention_geometry_enabled=False,
    )
    run_id = build_semantic_watermark_run_id(config)
    run_dir = tmp_path / config.output_dir / run_id
    run_dir.mkdir(parents=True)
    files = {
        "clean_image_path": run_dir / "clean_image.png",
        "watermarked_image_path": run_dir / "watermarked_image.png",
        "update_record_path": run_dir / "latent_update_records.jsonl",
        "detection_record_path": run_dir / "image_only_detection_records.jsonl",
    }
    for path in files.values():
        path.write_bytes(b"fixture")
    manifest_path = run_dir / "manifest.local.json"
    result_path = run_dir / "runtime_result.json"
    config_payload = semantic_watermark_runtime_config_payload(config)
    config_digest = build_stable_digest(config_payload)
    result_payload = {
        "run_id": run_id,
        "run_decision": "pass",
        **{key: path.relative_to(tmp_path).as_posix() for key, path in files.items()},
        "manifest_path": manifest_path.relative_to(tmp_path).as_posix(),
        "update_count": 1,
        "clean_detection_positive": False,
        "watermarked_detection_positive": True,
        "elapsed_seconds": 1.0,
        "metadata": {
            "scientific_unit_config": config_payload,
            "scientific_unit_provenance": (
                build_test_scientific_unit_provenance(
                    run_id,
                    config_digest,
                )
            ),
        },
    }
    result_path.write_text(json.dumps(result_payload), encoding="utf-8")
    output_paths = [path.relative_to(tmp_path).as_posix() for path in files.values()]
    output_paths.extend((result_path.relative_to(tmp_path).as_posix(), manifest_path.relative_to(tmp_path).as_posix()))
    manifest_path.write_text(
        json.dumps(
            {
                "config_digest": config_digest,
                "code_version": resolve_code_version(tmp_path),
                "output_paths": output_paths,
            }
        ),
        encoding="utf-8",
    )

    cached = load_completed_semantic_watermark_runtime_result(config, root=tmp_path)
    assert cached is not None
    assert cached.run_id == run_id

    files["clean_image_path"].unlink()
    assert load_completed_semantic_watermark_runtime_result(config, root=tmp_path) is None


@pytest.mark.quick
def test_closed_archive_recovery_without_directories_is_empty(
    tmp_path: Path,
) -> None:
    """未配置外部归档目录时恢复路径必须保持无操作."""

    recovered = scientific_workflow._recover_closed_archives(
        root_path=tmp_path,
        paper_run_name="probe_paper",
        target_fpr=0.1,
        expected_roles={
            "image_only_dataset_runtime",
            "dataset_level_quality",
        },
        archive_destination_dirs=None,
    )

    assert recovered["recovered_roles"] == []
    assert recovered["local_archives"] == {}
    assert recovered["all_expected_roles_recovered"] is False


@pytest.mark.quick
def test_partial_closed_archive_recovery_neither_extracts_nor_skips_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """只恢复主方法包时不得提取旧结果或跳过当前科学子命令."""

    run_name = "probe_paper"
    runtime_candidate = write_recovery_candidate(
        tmp_path,
        "image_only_dataset_runtime",
    )
    quality_dir = tmp_path / "drive" / "dataset_level_quality"
    quality_dir.mkdir(parents=True)
    progress_path = (
        tmp_path
        / "outputs"
        / "image_only_dataset_runtime"
        / run_name
        / "dataset_runtime_progress.json"
    )
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(
        json.dumps(
            {
                "protocol_decision": "resume_required",
                "remaining_prompt_count": 65,
            }
        ),
        encoding="utf-8",
    )
    execution_path = tmp_path / "outputs" / "scientific_execution.json"
    execution_path.write_text("{}", encoding="utf-8")
    execution_report = {
        "decision": "pass",
        "failure_reasons": [],
        "profile_id": scientific_workflow.SCIENTIFIC_PROFILE_ID,
        "profile_digest": "1" * 64,
        "complete_hash_lock_digest": "2" * 64,
        "dependency_environment_report_path": str(execution_path),
        "dependency_environment_report_digest": "3" * 64,
    }
    execution_calls: list[tuple[object, ...]] = []
    extraction_calls: list[Path] = []

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", run_name)
    monkeypatch.setattr(
        scientific_workflow,
        "inspect_closure_package",
        lambda package_path, **_kwargs: runtime_candidate,
    )
    monkeypatch.setattr(
        scientific_workflow,
        "_candidate_matches_repository",
        lambda _candidate, _root: True,
    )
    monkeypatch.setattr(
        scientific_workflow,
        "_extract_validated_archive",
        lambda package_path, _root: extraction_calls.append(package_path),
    )

    def execute_once(*args: object, **_kwargs: object) -> tuple[dict[str, object], Path]:
        """记录部分恢复后仍实际调用隔离科学命令."""

        execution_calls.append(args)
        return execution_report, execution_path

    monkeypatch.setattr(
        scientific_workflow,
        "execute_isolated_scientific_command",
        execute_once,
    )
    monkeypatch.setattr(
        scientific_workflow,
        "validate_scientific_execution_report",
        lambda *_args, **_kwargs: execution_report,
    )

    summary = scientific_workflow.run_semantic_watermark_image_only_session(
        tmp_path,
        archive_destination_dirs={
            "image_only_dataset_runtime": runtime_candidate.package_path.parent,
            "dataset_level_quality": quality_dir,
        },
    )

    assert len(execution_calls) == 1
    assert extraction_calls == []
    assert summary["workflow_decision"] == "resume_required"
    assert summary["closed_archive_recovery_ready"] is False
    assert summary["closed_archive_recovery"]["recovered_roles"] == [
        "image_only_dataset_runtime"
    ]


@pytest.mark.quick
def test_all_current_closed_archives_restore_without_new_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全部请求角色通过当前身份校验后才允许整体恢复并结束会话."""

    candidates = {
        role: write_recovery_candidate(tmp_path, role)
        for role in (
            "image_only_dataset_runtime",
            "dataset_level_quality",
        )
    }
    candidate_by_path = {
        candidate.package_path.resolve(): candidate
        for candidate in candidates.values()
    }
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "probe_paper")
    monkeypatch.setattr(
        scientific_workflow,
        "inspect_closure_package",
        lambda package_path, **_kwargs: candidate_by_path[package_path.resolve()],
    )
    monkeypatch.setattr(
        scientific_workflow,
        "_candidate_matches_repository",
        lambda _candidate, _root: True,
    )

    def reject_execution(*_args: object, **_kwargs: object) -> object:
        """完整恢复后不允许创建伪造的当前科学执行."""

        raise AssertionError("全部闭合包已恢复时不应重新执行科学命令")

    monkeypatch.setattr(
        scientific_workflow,
        "execute_isolated_scientific_command",
        reject_execution,
    )
    summary = scientific_workflow.run_semantic_watermark_image_only_session(
        tmp_path,
        archive_destination_dirs={
            role: candidate.package_path.parent
            for role, candidate in candidates.items()
        },
    )

    assert summary["workflow_decision"] == "closed_archives_recovered"
    assert summary["closed_archive_recovery_ready"] is True
    assert set(summary["recovered_roles"]) == set(candidates)
    assert set(summary["local_archives"]) == set(candidates)
    assert all(
        Path(path).is_file() for path in summary["local_archives"].values()
    )
    assert (
        tmp_path
        / "outputs"
        / "image_only_dataset_runtime"
        / "probe_paper"
        / "recovered_image_only_dataset_runtime.json"
    ).is_file()


@pytest.mark.quick
@pytest.mark.parametrize("drift_kind", ["code_version", "dependency_lock"])
def test_closed_archive_candidate_rejects_repository_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_kind: str,
) -> None:
    """旧提交或科学依赖锁漂移的闭合包不得视为当前可恢复结果."""

    candidate = write_recovery_candidate(
        tmp_path,
        "image_only_dataset_runtime",
    )
    execution_lock = {
        "formal_execution_commit": candidate.code_version,
        "formal_execution_lock_digest": (
            candidate.formal_execution_run_lock_digest
        ),
    }
    profile = SimpleNamespace(
        profile_name=candidate.scientific_profile_id,
        profile_digest=candidate.scientific_profile_digest,
        direct_requirements_digest=(
            candidate.scientific_direct_requirements_digest
        ),
        complete_hash_lock_digest=(
            candidate.scientific_complete_hash_lock_digest
        ),
        complete_hash_lock_dependency_count=(
            candidate.scientific_complete_hash_lock_dependency_count
        ),
        formal_ready=True,
        readiness_blockers=(),
    )
    monkeypatch.setattr(
        scientific_workflow,
        "require_published_formal_execution_lock",
        lambda _root: execution_lock,
    )
    monkeypatch.setattr(
        scientific_workflow,
        "require_dependency_profile_ready",
        lambda _profile_id, _registry_path: profile,
    )

    assert scientific_workflow._candidate_matches_repository(
        candidate,
        tmp_path,
    ) is True
    if drift_kind == "code_version":
        candidate.code_version = "f" * 40
    else:
        candidate.scientific_complete_hash_lock_digest = "f" * 64

    assert scientific_workflow._candidate_matches_repository(
        candidate,
        tmp_path,
    ) is False


@pytest.mark.quick
def test_closed_archive_recovery_rejects_same_time_different_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一生成时间出现两个不同摘要时不得任意选择一个包."""

    generated_at_utc = datetime(2026, 7, 12, tzinfo=timezone.utc)
    candidates = [
        write_recovery_candidate(
            tmp_path,
            "image_only_dataset_runtime",
            suffix=suffix,
            generated_at_utc=generated_at_utc,
        )
        for suffix in ("20260712t000000z_a", "20260712t000000z_b")
    ]
    candidate_by_path = {
        candidate.package_path.resolve(): candidate for candidate in candidates
    }
    monkeypatch.setattr(
        scientific_workflow,
        "inspect_closure_package",
        lambda package_path, **_kwargs: candidate_by_path[package_path.resolve()],
    )
    monkeypatch.setattr(
        scientific_workflow,
        "_candidate_matches_repository",
        lambda _candidate, _root: True,
    )

    with pytest.raises(RuntimeError, match="同时间不同内容"):
        scientific_workflow._recover_closed_archives(
            root_path=tmp_path,
            paper_run_name="probe_paper",
            target_fpr=0.1,
            expected_roles={"image_only_dataset_runtime"},
            archive_destination_dirs={
                "image_only_dataset_runtime": candidates[0].package_path.parent,
            },
        )


@pytest.mark.quick
def test_closed_archive_extraction_rejects_path_escape(tmp_path: Path) -> None:
    """ZIP 成员即使包含父目录跳转也不得写出 outputs 边界."""

    package_path = tmp_path / "outputs" / "malicious.zip"
    package_path.parent.mkdir(parents=True)
    with ZipFile(package_path, "w") as archive:
        archive.writestr("outputs/safe.json", "{}")
        archive.writestr("outputs/../escaped.json", "{}")

    with pytest.raises(ValueError):
        scientific_workflow._extract_validated_archive(
            package_path,
            tmp_path,
        )
    assert not (tmp_path / "escaped.json").exists()
    assert not (tmp_path / "outputs" / "safe.json").exists()


@pytest.mark.quick
def test_colab_image_only_session_reports_persistent_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colab 调度器必须优先返回主流程续跑状态, 不能误读旧正式摘要。"""

    run_name = "probe_paper"
    output_dir = tmp_path / "outputs" / "image_only_dataset_runtime" / run_name
    output_dir.mkdir(parents=True)
    (output_dir / "dataset_runtime_progress.json").write_text(
        json.dumps({"protocol_decision": "resume_required", "remaining_prompt_count": 65}),
        encoding="utf-8",
    )
    (output_dir / "dataset_runtime_summary.json").write_text(
        json.dumps({"protocol_decision": "pass"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", run_name)
    execution_path = tmp_path / "outputs" / "scientific_execution.json"
    execution_path.write_text("{}", encoding="utf-8")
    execution_report = {
        "decision": "pass",
        "failure_reasons": [],
        "profile_id": "sd35_method_runtime_gpu",
        "profile_digest": "1" * 64,
        "complete_hash_lock_digest": "2" * 64,
        "dependency_environment_report_path": str(execution_path),
        "dependency_environment_report_digest": "3" * 64,
    }
    execution_calls = []

    def execute_once(*args: object, **kwargs: object) -> tuple[dict[str, object], Path]:
        execution_calls.append((args, kwargs))
        return execution_report, execution_path

    monkeypatch.setattr(scientific_workflow, "execute_isolated_scientific_command", execute_once)
    monkeypatch.setattr(
        scientific_workflow,
        "validate_scientific_execution_report",
        lambda *args, **kwargs: execution_report,
    )

    summary = scientific_workflow.run_semantic_watermark_image_only_session(tmp_path)

    assert len(execution_calls) == 1
    assert execution_calls[0][0][0] == "sd35_method_runtime_gpu"
    assert summary["workflow_decision"] == "resume_required"
    assert summary["active_workflow"] == "image_only_dataset_runtime"
    assert summary["runtime_progress"]["remaining_prompt_count"] == 65


@pytest.mark.quick
def test_colab_image_only_session_mirrors_completed_formal_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主流程完成后应把盲检包和正式质量包镜像到论文运行目录。"""

    run_name = "probe_paper"
    runtime_dir = tmp_path / "outputs" / "image_only_dataset_runtime" / run_name
    quality_dir = tmp_path / "outputs" / "dataset_level_quality" / run_name
    runtime_dir.mkdir(parents=True)
    quality_dir.mkdir(parents=True)
    (runtime_dir / "dataset_runtime_summary.json").write_text(
        json.dumps({"protocol_decision": "pass", "supports_paper_claim": True}),
        encoding="utf-8",
    )
    (quality_dir / "dataset_quality_summary.json").write_text(
        json.dumps(
            {
                "formal_fid_kid_ready": True,
                "formal_fid_kid_claim_gate_ready": True,
                "canonical_formal_feature_extractor_ready": True,
                "supports_paper_claim": True,
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "image_only_dataset_runtime_package_fixture.zip").write_bytes(b"runtime")
    (quality_dir / "dataset_level_quality_package_fixture.zip").write_bytes(b"quality")
    runtime_drive_dir = tmp_path / "drive" / "image_only_dataset_runtime"
    quality_drive_dir = tmp_path / "drive" / "dataset_level_quality"
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", run_name)
    monkeypatch.setenv("SLM_WM_IMAGE_ONLY_RUNTIME_DRIVE_DIR", str(runtime_drive_dir))
    monkeypatch.setenv("SLM_WM_DATASET_QUALITY_DRIVE_DIR", str(quality_drive_dir))
    execution_path = tmp_path / "outputs" / "scientific_execution.json"
    execution_path.write_text("{}", encoding="utf-8")
    execution_report = {
        "decision": "pass",
        "failure_reasons": [],
        "profile_id": "sd35_method_runtime_gpu",
        "profile_digest": "1" * 64,
        "complete_hash_lock_digest": "2" * 64,
        "dependency_environment_report_path": str(execution_path),
        "dependency_environment_report_digest": "3" * 64,
    }
    execution_calls = []

    def execute_once(*args: object, **kwargs: object) -> tuple[dict[str, object], Path]:
        execution_calls.append((args, kwargs))
        return execution_report, execution_path

    monkeypatch.setattr(scientific_workflow, "execute_isolated_scientific_command", execute_once)
    monkeypatch.setattr(
        scientific_workflow,
        "validate_scientific_execution_report",
        lambda *args, **kwargs: execution_report,
    )
    monkeypatch.setattr(scientific_workflow, "_write_bindings", lambda **kwargs: {})
    monkeypatch.setattr(scientific_workflow, "_run_bound_packaging", lambda **kwargs: {})
    monkeypatch.setattr(
        scientific_workflow,
        "_archive_paths_from_packaging",
        lambda *args, **kwargs: {
            "image_only_dataset_runtime": runtime_dir
            / "image_only_dataset_runtime_package_fixture.zip",
            "dataset_level_quality": quality_dir
            / "dataset_level_quality_package_fixture.zip",
        },
    )

    summary = scientific_workflow.run_semantic_watermark_image_only_session(
        tmp_path,
        archive_destination_dirs={
            "image_only_dataset_runtime": runtime_drive_dir,
            "dataset_level_quality": quality_drive_dir,
        },
    )

    assert len(execution_calls) == 1
    assert summary["workflow_decision"] == "dataset_complete"
    assert (runtime_drive_dir / "image_only_dataset_runtime_package_fixture.zip").read_bytes() == b"runtime"
    assert (quality_drive_dir / "dataset_level_quality_package_fixture.zip").read_bytes() == b"quality"


@pytest.mark.quick
def test_formal_ablation_resume_skips_binding_and_packaging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式消融仍有 progress 时不得重复生成主运行与质量归档."""

    run_name = "probe_paper"
    runtime_dir = tmp_path / "outputs" / "image_only_dataset_runtime" / run_name
    quality_dir = tmp_path / "outputs" / "dataset_level_quality" / run_name
    ablation_dir = tmp_path / "outputs" / "formal_mechanism_ablation" / run_name
    for output_dir in (runtime_dir, quality_dir, ablation_dir):
        output_dir.mkdir(parents=True)
    (runtime_dir / "dataset_runtime_summary.json").write_text(
        json.dumps({"protocol_decision": "pass"}),
        encoding="utf-8",
    )
    (quality_dir / "dataset_quality_summary.json").write_text(
        json.dumps({"formal_fid_kid_claim_gate_ready": True}),
        encoding="utf-8",
    )
    (ablation_dir / "runtime_rerun_progress.json").write_text(
        json.dumps(
            {
                "protocol_decision": "resume_required",
                "remaining_run_count": 555,
            }
        ),
        encoding="utf-8",
    )
    execution_path = tmp_path / "outputs" / "scientific_execution.json"
    execution_path.write_text("{}", encoding="utf-8")
    execution_report = {
        "decision": "pass",
        "failure_reasons": [],
        "profile_id": "sd35_method_runtime_gpu",
        "profile_digest": "1" * 64,
        "complete_hash_lock_digest": "2" * 64,
        "dependency_environment_report_path": str(execution_path),
        "dependency_environment_report_digest": "3" * 64,
    }
    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", run_name)
    monkeypatch.setattr(
        scientific_workflow,
        "execute_isolated_scientific_command",
        lambda *args, **kwargs: (execution_report, execution_path),
    )
    monkeypatch.setattr(
        scientific_workflow,
        "validate_scientific_execution_report",
        lambda *args, **kwargs: execution_report,
    )

    def reject_packaging(**kwargs: object) -> object:
        raise AssertionError("消融续跑状态不得写 binding 或执行打包")

    monkeypatch.setattr(scientific_workflow, "_write_bindings", reject_packaging)
    monkeypatch.setattr(scientific_workflow, "_run_bound_packaging", reject_packaging)

    summary = scientific_workflow.run_semantic_watermark_image_only_session(
        tmp_path,
        run_formal_ablation=True,
    )

    assert summary["workflow_decision"] == "resume_required"
    assert summary["active_workflow"] == "runtime_rerun_ablation"
    assert summary["ablation_progress"]["remaining_run_count"] == 555
    assert "local_archives" not in summary
    assert "scientific_execution_bindings" not in summary


@pytest.mark.quick
def test_bound_packaging_archive_roles_must_match_exact_requested_set(
    tmp_path: Path,
) -> None:
    """绑定打包结果必须无重复且精确覆盖当前请求的产物角色."""

    runtime_archive = tmp_path / "outputs" / "runtime.zip"
    quality_archive = tmp_path / "outputs" / "quality.zip"
    runtime_archive.parent.mkdir(parents=True)
    runtime_archive.write_bytes(b"runtime")
    quality_archive.write_bytes(b"quality")

    def record(role: str, path: Path) -> dict[str, object]:
        return {
            "artifact_role": role,
            "archive_path": path.relative_to(tmp_path).as_posix(),
            "archive_sha256": scientific_workflow.file_sha256(path),
        }

    expected_roles = {
        "image_only_dataset_runtime",
        "dataset_level_quality",
    }
    valid_execution = {
        "packaging_result": {
            "archives": [
                record("image_only_dataset_runtime", runtime_archive),
                record("dataset_level_quality", quality_archive),
            ]
        }
    }
    resolved = scientific_workflow._archive_paths_from_packaging(
        tmp_path,
        valid_execution,
        expected_roles=expected_roles,
    )
    assert set(resolved) == expected_roles

    missing_execution = {
        "packaging_result": {
            "archives": [record("image_only_dataset_runtime", runtime_archive)]
        }
    }
    with pytest.raises(RuntimeError, match="角色集合不一致"):
        scientific_workflow._archive_paths_from_packaging(
            tmp_path,
            missing_execution,
            expected_roles=expected_roles,
        )

    duplicate_execution = {
        "packaging_result": {
            "archives": [
                record("image_only_dataset_runtime", runtime_archive),
                record("image_only_dataset_runtime", runtime_archive),
            ]
        }
    }
    with pytest.raises(RuntimeError, match="重复角色"):
        scientific_workflow._archive_paths_from_packaging(
            tmp_path,
            duplicate_execution,
            expected_roles=expected_roles,
        )
