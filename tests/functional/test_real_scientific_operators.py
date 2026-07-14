"""验证真实风险、Jacobian、载体、注意力和仅图像检测算子。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import math
from dataclasses import MISSING, replace
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
    LowFrequencyCarrierConfig,
    build_low_frequency_template,
    build_tail_robust_template,
    compute_blind_content_score,
    keyed_prg_protocol_record,
    tail_robust_carrier_protocol_record,
)
from main.methods.detection import (
    ImageOnlyDetectionConfig,
    detect_image_only_watermark,
    recompute_image_only_detection_digest_payload,
    validate_image_only_detection_digest_record,
)
from main.methods.geometry import (
    ATTENTION_ALIGNMENT_ANCHOR_COUNT,
    ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO,
    ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD,
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    ATTENTION_IMAGE_PADDING_MODE,
    ATTENTION_IMAGE_QUANTIZATION_PROTOCOL,
    ATTENTION_IMAGE_RESAMPLING_MODE,
    ATTENTION_RELATION_COMPONENT_NAMES,
    DifferentiableAttentionRecorder,
    FROZEN_SD35_ATTENTION_MODULE_NAMES,
    QKAttentionRelation,
    StableAttentionTokenSelection,
    attention_geometry_score,
    attention_relation_component_protocol,
    attention_relation_component_scores,
    attention_relation_stability_map,
    build_attention_relation_descriptor,
    build_attention_relation_graph_identity,
    build_qk_atomic_content_metadata,
    build_stable_attention_pair_weights,
    combine_attention_relation_component_scores,
    compute_attention_geometry_gradient,
    keyed_attention_relation_projection,
    optimize_attention_geometry_update,
    qk_atomic_content_records_digest,
    qk_atomic_evaluation_records_digest,
    qk_operator_metadata_records_digest,
    qk_self_attention,
    recompute_attention_alignment_digest_payload,
    recover_attention_affine_alignment,
    select_stable_attention_tokens,
)
from main.methods.geometry.differentiable_attention import keyed_relation_signs
from main.methods.semantic import build_branch_risk_fields
from main.methods.subspace import (
    JACOBIAN_NULL_SPACE_EVIDENCE_VERSION,
    JacobianNullSpaceResult,
    build_exact_jacobian_linearization,
    exact_jvp,
    generate_keyed_candidate_directions,
    recompute_jacobian_null_space_result_digest,
    solve_jacobian_null_space,
    solve_psd_conjugate_gradient,
)
from main.methods.update_composition import (
    QUANTIZED_COMPOSITION_EVIDENCE_VERSION,
    RiskBoundedUpdate,
    compose_ordered_float32_update_once,
    recompute_quantized_composition_evidence_digest,
)
from experiments.runners.image_only_dataset_runtime import (
    FrozenEvidenceProtocol,
    _decision,
    _scientific_update_record_ready,
    apply_frozen_evidence_protocol,
    calibrate_complete_evidence_protocol,
    frozen_evidence_protocol_digest_payload,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    _align_image,
    _attention_modules,
    _bind_public_detection_noise_qk_evidence,
    _branch_risk_configs,
    _image_attention_extractor,
    _public_detection_noise_prg_identity,
    _public_detection_noise_tensor,
    _public_detection_noise_seed,
    _post_risk_direction_jacobian_record,
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
from tests.helpers.formal_detection_record import bind_formal_detection_record


_FORMAL_ATTENTION_ALIGNMENT_GATE = {
    "attention_anchor_count": ATTENTION_ALIGNMENT_ANCHOR_COUNT,
    "attention_residual_threshold": ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD,
    "attention_minimum_inlier_ratio": (
        ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO
    ),
}
_FORMAL_LOW_FREQUENCY_CONFIG = LowFrequencyCarrierConfig(
    kernel_size=5,
    stride=1,
    padding=2,
    boundary_mode="zero_padding",
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
)
_FORMAL_LOW_FREQUENCY_PROTOCOL = (
    _FORMAL_LOW_FREQUENCY_CONFIG.to_record()
)
_FORMAL_LF_WEIGHT = 0.70
_FORMAL_TAIL_ROBUST_WEIGHT = 0.30
_FORMAL_TAIL_FRACTION = 0.20
_FORMAL_TAIL_CARRIER_PROTOCOL = tail_robust_carrier_protocol_record(
    _FORMAL_TAIL_FRACTION,
    prg_version=KEYED_PRG_VERSION,
)


def _formal_content_carrier_identity_fields() -> dict[str, object]:
    """返回检测正文和 metadata 共享的完整内容载体身份."""

    return {
        "lf_carrier_protocol_digest": (
            _FORMAL_LOW_FREQUENCY_CONFIG.protocol_digest
        ),
        "lf_weight": _FORMAL_LF_WEIGHT,
        "tail_robust_weight": _FORMAL_TAIL_ROBUST_WEIGHT,
        "tail_fraction": _FORMAL_TAIL_FRACTION,
        "tail_carrier_protocol_digest": (
            _FORMAL_TAIL_CARRIER_PROTOCOL[
                "tail_carrier_protocol_digest"
            ]
        ),
    }


def _fixture_keyed_tensor_carrier_identity(
    branch_name: str,
    carrier_protocol_digest: str,
    projection_energy_retention: float,
    *,
    null_space_digest: str = "1" * 64,
) -> dict[str, object]:
    """构造不持久化嵌套摘要正文的固定载体测试引用."""

    payload: dict[str, object] = {
        "branch_name": branch_name,
        "template_shape": [1, 16, 64, 64],
        "projection_energy_retention": round(
            projection_energy_retention,
            12,
        ),
        "minimum_projection_energy_retention": 0.01,
        "null_space_digest": null_space_digest,
        "canonical_template_content_sha256": "2" * 64,
        "embedded_direction_content_sha256": "3" * 64,
        "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
        "keyed_prg_version": KEYED_PRG_VERSION,
        "keyed_prg_protocol_digest": keyed_prg_protocol_record()[
            "keyed_prg_protocol_digest"
        ],
        "carrier_protocol_digest": carrier_protocol_digest,
    }
    return {
        "branch_name": branch_name,
        "projection_energy_retention": projection_energy_retention,
        "carrier_protocol_digest": carrier_protocol_digest,
        "template_shape": [1, 16, 64, 64],
        "canonical_template_content_sha256": "2" * 64,
        "template_digest": build_stable_digest(payload),
    }


def _image_only_detection_config(
    **overrides: object,
) -> ImageOnlyDetectionConfig:
    """构造显式绑定全部内容协议字段的轻量盲检配置."""

    values: dict[str, object] = {
        "model_id": "model",
        "attention_module_names": FROZEN_SD35_ATTENTION_MODULE_NAMES,
        "content_threshold": 0.20,
        "geometry_score_threshold": 0.0,
        "attention_anchor_count": ATTENTION_ALIGNMENT_ANCHOR_COUNT,
        "attention_residual_threshold": ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD,
        "attention_minimum_inlier_ratio": (
            ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO
        ),
        "low_frequency_config": _FORMAL_LOW_FREQUENCY_CONFIG,
        "lf_weight": _FORMAL_LF_WEIGHT,
        "tail_robust_weight": _FORMAL_TAIL_ROBUST_WEIGHT,
        "tail_fraction": _FORMAL_TAIL_FRACTION,
        "keyed_prg_version": KEYED_PRG_VERSION,
        "registration_confidence_threshold": 0.0,
        "attention_sync_score_threshold": 0.0,
        "rescue_margin_low": -0.05,
        "attention_stable_token_fraction": 0.50,
        "attention_unstable_pair_weight": 0.25,
        "attention_relation_component_weights": (
            0.25,
            0.25,
            0.25,
            0.25,
        ),
    }
    values.update(overrides)
    return ImageOnlyDetectionConfig(**values)


def _formal_detection_alignment_identity(
    *,
    registration_geometry_reliable: bool,
    geometry_reliable: bool | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """构造与生产记录相同的注意力结构门禁身份字段."""

    resolved_geometry_reliable = (
        registration_geometry_reliable
        if geometry_reliable is None
        else geometry_reliable
    )
    gate = dict(_FORMAL_ATTENTION_ALIGNMENT_GATE)
    detector_metadata = {
        "attention_alignment_gate": dict(gate),
        "stable_pair_weight_identity_ready": True,
        **gate,
        **_formal_content_carrier_identity_fields(),
    }
    alignment = {
        "registration_geometry_reliable": registration_geometry_reliable,
        "geometry_reliable": resolved_geometry_reliable,
        "metadata": {"attention_alignment_gate": dict(gate)},
        **gate,
    }
    return detector_metadata, alignment


@pytest.mark.quick
def test_image_only_detection_config_has_no_method_parameter_defaults() -> None:
    """盲检全部科学参数必须由正式运行配置显式提供."""

    config_fields = ImageOnlyDetectionConfig.__dataclass_fields__
    for field_name in config_fields:
        field = config_fields[field_name]
        assert field.default is MISSING
        assert field.default_factory is MISSING

    decision_parameters = inspect.signature(_decision).parameters
    for field_name in (
        "geometry_score_threshold",
        "registration_confidence_threshold",
        "attention_sync_score_threshold",
    ):
        assert decision_parameters[field_name].default is inspect.Parameter.empty


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


def write_formal_method_config(root: Path) -> None:
    """把正式方法 YAML 显式复制到隔离 workflow 测试根目录。"""

    source = Path(__file__).resolve().parents[2] / "configs" / "model_sd35.yaml"
    target = root / "configs" / "model_sd35.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


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
        model_id="Manojb/stable-diffusion-2-1-base",
        model_revision="0094d483a120f3f33dafbd187ea4aa60d10de75c",
        width=base.width,
        height=base.height,
        inference_steps=base.inference_steps,
        public_detection_schedule_index=base.public_detection_schedule_index,
        public_detection_noise_prg_protocol=(
            base.public_detection_noise_prg_protocol
        ),
        public_detection_noise_domain=base.public_detection_noise_domain,
        public_detection_conditioning_protocol=(
            base.public_detection_conditioning_protocol
        ),
        public_detection_condition_text=base.public_detection_condition_text,
    )

    assert _public_detection_noise_seed(base) == _public_detection_noise_seed(changed_sample)
    assert _public_detection_noise_seed(base) != _public_detection_noise_seed(changed_model)


@pytest.mark.quick
def test_public_detection_noise_uses_canonical_gaussian_prg() -> None:
    """公开检测噪声必须跨生成样本稳定, 并绑定实际 dtype Tensor 字节。"""

    base = SemanticWatermarkRuntimeConfig()
    changed_sample = replace(
        base,
        seed=base.seed + 17,
        prompt="另一条生成 Prompt",
        prompt_id="other_prompt",
    )
    latent = torch.zeros((1, 4, 2, 2), dtype=torch.float16)

    first = _public_detection_noise_tensor(latent, base)
    second = _public_detection_noise_tensor(latent, changed_sample)

    assert first.device.type == "cpu"
    assert first.dtype == torch.float16
    assert torch.equal(first, second)
    assert tensor_content_sha256(first) == tensor_content_sha256(second)
    assert tensor_content_sha256(first) == (
        "15c7795fcd10c030a8ec13c0fc4478b276f1d0bf52108f97c484dc4495f01988"
    )
    identity = _public_detection_noise_prg_identity(
        base,
        tuple(int(value) for value in latent.shape),
    )
    assert identity["key_material"] == base.public_detection_noise_domain
    assert identity["keyed_prg_protocol_digest"] == (
        "a6266dc1fb4a59f8038062dcd120f145582153138b8176baae12013d5a22687b"
    )
    assert identity["domain_fields"] == {
        "operator": base.public_detection_noise_domain,
        "model_id": base.model_id,
        "model_revision": base.model_revision,
        "width": 512,
        "height": 512,
        "inference_steps": 20,
        "public_detection_schedule_index": 7,
        "latent_shape": (1, 4, 2, 2),
    }
    assert identity["public_detection_noise_prg_identity_digest"] == (
        "87e5a8b39005a9f9ad9f5e822e4a56fad056e87df9201b31f345358c19b7f5c3"
    )
    source = inspect.getsource(_public_detection_noise_tensor)
    assert "build_keyed_gaussian_tensor" in source
    assert "torch.randn" not in source
    assert "torch.Generator" not in source


@pytest.mark.quick
def test_branch_risk_fields_use_opposite_texture_preferences() -> None:
    """LF 应回避高纹理, 尾部鲁棒分支应偏好高纹理。"""

    runtime_config = SemanticWatermarkRuntimeConfig()
    fields = build_branch_risk_fields(
        semantic_values=(0.2, 0.2),
        texture_values=(0.1, 0.9),
        adjacent_step_stability_values=(0.8, 0.8),
        local_contrast_risk_values=(0.2, 0.2),
        attention_stability_values=(0.8, 0.8),
        configs=_branch_risk_configs(runtime_config),
        risk_neutral_texture_value=(
            runtime_config.risk_neutral_texture_value
        ),
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
    assert result.basis_response_matrix.requires_grad is False
    assert abs(float(result.latent_basis[3, 0])) == pytest.approx(1.0, abs=1e-6)
    assert result.metadata["solver"] == "matrix_free_full_jacobian_psd_cg"
    assert result.metadata["cg_damping"] == 0.0
    assert result.to_record()["latent_basis_content_sha256"] == (
        tensor_content_sha256(result.latent_basis)
    )
    assert result.to_record()["basis_response_matrix_content_sha256"] == (
        tensor_content_sha256(result.basis_response_matrix)
    )
    assert result.to_record()[
        "projected_direction_matrix_content_sha256"
    ] == tensor_content_sha256(result.projected_direction_matrix)
    assert result.to_record()[
        "projected_direction_response_matrix_content_sha256"
    ] == tensor_content_sha256(result.projected_direction_response_matrix)
    assert result.to_record()[
        "basis_reference_response_matrix_content_sha256"
    ] == tensor_content_sha256(result.basis_reference_response_matrix)
    assert result.column_reference_response_norms == pytest.approx(
        tuple(
            float(torch.linalg.norm(column).item())
            for column in result.basis_reference_response_matrix.unbind(dim=1)
        )
    )
    solver_record = result.to_record()
    assert recompute_jacobian_null_space_result_digest(
        solver_record
    ) == result.solver_digest
    solver_record["column_response_norms"][0] = 0.25
    assert recompute_jacobian_null_space_result_digest(
        solver_record
    ) != result.solver_digest


@pytest.mark.quick
def test_null_projection_energy_retention_uses_squared_l2_ratio() -> None:
    """Null Space 投影保留率必须比较平方 L2 能量, 而不是振幅范数。"""

    latent = torch.zeros(2)

    def full_features(values: torch.Tensor) -> torch.Tensor:
        return values[:1]

    inverse_square_root_two = 2.0**-0.5
    candidate = torch.tensor(
        ((inverse_square_root_two,), (inverse_square_root_two,))
    )
    result = solve_jacobian_null_space(
        latent=latent,
        candidate_matrix=candidate,
        risk_budget=torch.ones_like(latent),
        null_rank=1,
        joint_feature_linearization=build_exact_jacobian_linearization(
            full_features,
            latent,
        ),
        branch_name="lf_content",
    )

    # 候选向量能量为1, 投影后只保留第二个坐标, 能量为1/2。
    assert result.projection_energy_retentions == pytest.approx((0.5,))
    # 范数比例为 sqrt(1/2), 此断言独立阻止实现退回振幅比例。
    assert result.projection_energy_retentions[0] != pytest.approx(
        inverse_square_root_two
    )


@pytest.mark.quick
def test_qr_basis_uses_independent_routed_candidate_references() -> None:
    """QR 每列参考必须通过右侧三角求解与同一列混合保持一致。"""

    latent = torch.zeros(5)

    def full_features(values: torch.Tensor) -> torch.Tensor:
        return values[:2]

    candidate_matrix = torch.tensor(
        (
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (0.0, 2.0),
        )
    )
    linearization = build_exact_jacobian_linearization(full_features, latent)
    result = solve_jacobian_null_space(
        latent=latent,
        candidate_matrix=candidate_matrix,
        risk_budget=torch.ones_like(latent),
        null_rank=2,
        joint_feature_linearization=linearization,
    )

    qr_factor = (
        result.latent_basis.transpose(0, 1)
        @ result.projected_direction_matrix
    )
    expected_reference = torch.linalg.solve_triangular(
        qr_factor.transpose(0, 1),
        result.routed_candidate_matrix.transpose(0, 1),
        upper=False,
    ).transpose(0, 1)
    expected_response = torch.stack(
        tuple(
            linearization.apply(expected_reference[:, index]).detach()
            for index in range(expected_reference.shape[1])
        ),
        dim=1,
    )

    assert torch.allclose(result.basis_reference_matrix, expected_reference)
    assert torch.allclose(
        result.basis_reference_response_matrix,
        expected_response,
    )
    assert result.column_reference_response_norms == pytest.approx(
        tuple(
            float(torch.linalg.norm(column).item())
            for column in expected_response.unbind(dim=1)
        )
    )


@pytest.mark.quick
def test_exact_jacobian_linearization_satisfies_adjoint_identity() -> None:
    """精确 JVP 与 VJP 必须满足同一 Jacobian 的伴随恒等式。"""

    latent = torch.tensor((0.2, -0.4, 0.7), dtype=torch.float32)

    def full_features(values: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                values[0] * values[1],
                values[1].square() + 2.0 * values[2],
            )
        )

    linearization = build_exact_jacobian_linearization(full_features, latent)
    direction = torch.tensor((0.3, -0.2, 0.5), dtype=torch.float32)
    tangent = linearization.apply(direction)
    cotangent = torch.tensor((0.6, -0.8), dtype=tangent.dtype)
    transpose_image = linearization.transpose_apply(cotangent)

    left = torch.dot(tangent, cotangent)
    right = torch.dot(direction.to(transpose_image.dtype), transpose_image)

    assert float(left.item()) == pytest.approx(float(right.item()), abs=1e-6)


@pytest.mark.quick
def test_post_risk_direction_reexecutes_independent_exact_jvp() -> None:
    """风险支持清理后的实际方向必须逐分支复验, 不能依赖联合抵消。"""

    latent = torch.zeros((1, 1, 1, 2), dtype=torch.float32)

    def feature_function(candidate: torch.Tensor) -> torch.Tensor:
        """只观测第一个 latent 坐标, 形成解析 Jacobian。"""

        return candidate[..., :1]

    record = _post_risk_direction_jacobian_record(
        feature_function=feature_function,
        latent=latent,
        branch_name="lf_content",
        actual_unit_direction=torch.tensor([[[[0.0, 1.0]]]]),
        reference_direction=torch.tensor([[[[1.0, 0.0]]]]),
        maximum_relative_response=1e-4,
        numerical_epsilon=1e-12,
    )

    assert record["branch_post_risk_jacobian_gate_ready"] is True
    assert record["branch_post_risk_relative_response_residual"] == 0.0
    assert record["branch_post_risk_direction_content_sha256"] == (
        tensor_content_sha256(torch.tensor([[[[0.0, 1.0]]]]))
    )
    with pytest.raises(RuntimeError, match="未通过精确 JVP 门禁"):
        _post_risk_direction_jacobian_record(
            feature_function=feature_function,
            latent=latent,
            branch_name="lf_content",
            actual_unit_direction=torch.tensor([[[[1.0, 0.0]]]]),
            reference_direction=torch.tensor([[[[1.0, 0.0]]]]),
            maximum_relative_response=1e-4,
            numerical_epsilon=1e-12,
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
def test_risk_budget_does_not_repeat_one_sample_across_batch() -> None:
    """多样本 latent 不得把单张 HxW 预算静默复制到其他样本。"""

    latent = torch.zeros((2, 1, 2, 2))

    def full_features(values: torch.Tensor) -> torch.Tensor:
        return values.reshape(-1)[:1]

    with pytest.raises(ValueError, match="axis_budget 无法广播"):
        solve_jacobian_null_space(
            latent=latent,
            candidate_matrix=torch.eye(latent.numel()),
            risk_budget=torch.ones(4),
            null_rank=1,
            joint_feature_linearization=build_exact_jacobian_linearization(
                full_features,
                latent,
            ),
        )


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
        "null_space_evidence_version": (
            JACOBIAN_NULL_SPACE_EVIDENCE_VERSION
        ),
        "branch_name": "lf_content",
        "basis_rank": 4,
        "null_rank": 4,
        "candidate_count": 20,
        "candidate_shape": [65536, 20],
        "evaluated_direction_count": 4,
        "evaluated_direction_indices": [0, 1, 2, 3],
        "response_width": JOINT_FEATURE_WIDTH,
        "response_shape": [JOINT_FEATURE_WIDTH, 4],
        "response_residual": 0.1,
        "relative_response_residual": 1e-6,
        "orthogonality_error": 1e-6,
        "column_response_norms": [1e-6] * 4,
        "column_relative_response_residuals": [1e-6] * 4,
        "column_reference_response_norms": [1.0] * 4,
        "projection_energy_retentions": [0.2] * 4,
        "cg_relative_residuals": [1e-7] * 4,
        "cg_iteration_counts": [1] * 4,
        "cg_converged": True,
        "metadata": {
            "jvp_mode": "torch_func_exact_jvp_vjp",
            "solver": "matrix_free_full_jacobian_psd_cg",
            "latent_basis_formula": (
                "qr(Bd - B^2 J^T solve_psd_cg(J B^2 J^T, J B d))"
            ),
            "full_feature_jvp": True,
            "full_feature_vjp": True,
            "cg_damping": 0.0,
            "cg_maximum_iterations": 64,
            "cg_relative_tolerance": 1e-6,
            "maximum_relative_response_residual": 1e-4,
            "minimum_projection_energy_retention": 0.01,
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
            "null_space_numerical_epsilon": 1e-12,
            "maximum_qr_condition_number": 1e6,
            "qr_condition_number": 1.0,
            "maximum_orthogonality_error": 1e-5,
            "qr_reference_solve_protocol": (
                "right_upper_triangular_solve_without_explicit_inverse_v1"
            ),
            "risk_budget_operator": "explicit_diagonal_B",
        },
        "candidate_matrix_content_sha256": "1" * 64,
        "risk_budget_content_sha256": "2" * 64,
        "routed_candidate_response_matrix_content_sha256": "3" * 64,
        "projected_direction_matrix_content_sha256": "4" * 64,
        "projected_direction_response_matrix_content_sha256": "5" * 64,
        "latent_basis_content_sha256": "4" * 64,
        "basis_response_matrix_content_sha256": "6" * 64,
        "basis_reference_response_matrix_content_sha256": "7" * 64,
    }
    config = SemanticWatermarkRuntimeConfig()
    tail_carrier_protocol_digest = build_stable_digest(
        {
            "tail_carrier_protocol_schema": (
                "slm_wm_tail_robust_carrier_protocol_v1"
            ),
            "tail_fraction": config.tail_fraction,
            "tail_selection_rule": (
                "descending_absolute_value_then_ascending_flat_index"
            ),
            "keyed_prg_version": config.keyed_prg_version,
        }
    )
    lf_template_identity = _fixture_keyed_tensor_carrier_identity(
        "lf_content",
        config.low_frequency_carrier_config.protocol_digest,
        0.2,
    )
    tail_template_identity = _fixture_keyed_tensor_carrier_identity(
        "tail_robust",
        tail_carrier_protocol_digest,
        0.2,
    )
    qk_role_scores = {
        "latent_before": 0.10,
        "optimization_content_base_latent": 0.11,
        "accepted_attention_candidate": 0.13,
        "actual_written_content_base_latent": 0.105,
        "actual_written_combined_latent": 0.12,
    }
    attention_qk_atomic_content_records = []
    for role_index, role in enumerate(qk_role_scores):
        role_values = torch.eye(4).reshape(1, 4, 4) * (
            0.1 + 0.01 * role_index
        )
        role_probabilities = torch.softmax(role_values, dim=-1)
        role_atoms = [
            build_qk_atomic_content_metadata(
                layer_name,
                role_values,
                role_values,
                role_values,
                role_probabilities,
                (0, 1, 2, 3),
            )
            for layer_name in config.attention_module_names
        ]
        attention_qk_atomic_content_records.append(
            {
                "qk_evaluation_role": role,
                "evaluation_latent_content_sha256": hashlib.sha256(
                    role.encode("utf-8")
                ).hexdigest(),
                "evaluation_score": qk_role_scores[role],
                "qk_atomic_content_records": role_atoms,
                "qk_atomic_content_digest": (
                    qk_atomic_content_records_digest(role_atoms)
                ),
                "qk_atomic_content_ready": True,
            }
        )
    qk_operator_metadata_records = [
        {
            "record_layer_name": layer_name,
            "module_layer_name": layer_name,
            "module_class_name": "FixtureAttention",
            "head_count": 2,
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
        for layer_name in config.attention_module_names
    ]
    null_space_records = {}
    for branch_name in ("lf_content", "tail_robust", "attention_geometry"):
        branch_subspace = {
            **subspace,
            "branch_name": branch_name,
            "metadata": dict(subspace["metadata"]),
        }
        branch_subspace["solver_digest"] = (
            recompute_jacobian_null_space_result_digest(
                branch_subspace
            )
        )
        null_space_records[branch_name] = branch_subspace
    lf_template_identity = _fixture_keyed_tensor_carrier_identity(
        "lf_content",
        config.low_frequency_carrier_config.protocol_digest,
        0.2,
        null_space_digest=null_space_records["lf_content"][
            "solver_digest"
        ],
    )
    tail_template_identity = _fixture_keyed_tensor_carrier_identity(
        "tail_robust",
        tail_carrier_protocol_digest,
        0.2,
        null_space_digest=null_space_records["tail_robust"][
            "solver_digest"
        ],
    )
    branch_update_content_records = {
        "lf_content": "5" * 64,
        "tail_robust": "6" * 64,
        "attention_geometry": "7" * 64,
    }
    branch_risk_records = {
        name: {
            "branch_name": name,
            "risk_field_digest": build_stable_digest(
                {"branch_name": name, "fixture": "risk_field"}
            ),
            "eligible_position_count": 10,
            "risk_values_content_sha256": "8" * 64,
            "budget_values_content_sha256": "9" * 64,
            "eligible_mask_content_sha256": "a" * 64,
            "effective_budget_values_content_sha256": "2" * 64,
            "branch_unit_direction_content_sha256": "d" * 64,
            "branch_budget_envelope_content_sha256": "b" * 64,
            "branch_written_update_content_sha256": (
                branch_update_content_records[name]
            ),
            "branch_post_risk_direction_content_sha256": "d" * 64,
            "branch_post_risk_reference_direction_content_sha256": "e" * 64,
            "branch_post_risk_response_content_sha256": "f" * 64,
            "branch_post_risk_reference_response_content_sha256": "0" * 64,
            "branch_post_risk_response_norm": 1e-5,
            "branch_post_risk_reference_response_norm": 1.0,
            "branch_post_risk_relative_response_residual": 1e-5,
            "branch_post_risk_jacobian_gate_ready": True,
            "branch_post_risk_jvp_mode": (
                "torch_autograd_exact_jvp_vjp_reexecution"
            ),
            "branch_nominal_strength": 0.01,
            "branch_applied_strength": 0.005,
            "branch_risk_scale_factor": 0.5,
            "branch_budget_ceiling": 1.0,
            "branch_direction_epsilon": (
                config.risk_bounded_scale_direction_epsilon
            ),
            "branch_numerical_epsilon": (
                config.null_space_numerical_epsilon
            ),
            "branch_written_update_maximum_envelope_ratio": 1.0,
        }
        for name in ("lf_content", "tail_robust", "attention_geometry")
    }
    branch_risk_content_records = {
        name: {
            field_name: branch_record[field_name]
            for field_name in (
                "risk_values_content_sha256",
                "budget_values_content_sha256",
                "eligible_mask_content_sha256",
                "effective_budget_values_content_sha256",
                "branch_unit_direction_content_sha256",
                "branch_budget_envelope_content_sha256",
                "branch_written_update_content_sha256",
                "branch_post_risk_direction_content_sha256",
                "branch_post_risk_reference_direction_content_sha256",
                "branch_post_risk_response_content_sha256",
                "branch_post_risk_reference_response_content_sha256",
            )
        }
        for name, branch_record in branch_risk_records.items()
    }
    record = {
        "step_index": 6,
        "scheduler_step_timestep": 6.0,
        "post_step_schedule_index": 7,
        "watermark_key_material_digest_random": "0" * 64,
        **_formal_content_carrier_identity_fields(),
        "tensor_content_digest_version": TENSOR_CONTENT_DIGEST_VERSION,
        "adjacent_step_reference_index": 5,
        "adjacent_step_reference_latent_content_sha256": "1" * 64,
        "adjacent_step_stability_status": (
            "measured_from_immediately_previous_scheduler_step"
        ),
        "latent_content_sha256_before": "a" * 64,
        "latent_content_sha256_after": "b" * 64,
        "current_decoded_rgb_content_sha256": "6" * 64,
        "previous_step_decoded_rgb_content_sha256": "7" * 64,
        "clip_patch_tokens_content_sha256": "8" * 64,
        "clip_cls_token_content_sha256": "9" * 64,
        "semantic_risk_signal_content_sha256": "1" * 64,
        "texture_risk_signal_content_sha256": "2" * 64,
        "local_contrast_risk_signal_content_sha256": "3" * 64,
        "adjacent_step_stability_signal_content_sha256": "4" * 64,
        "attention_stability_signal_content_sha256": "5" * 64,
        "active_carrier_branches": [
            "lf_content",
            "tail_robust",
            "attention_geometry",
        ],
        "branch_risk_bundle_digest": build_stable_digest(
            {
                name: branch_risk_records[name]["risk_field_digest"]
                for name in (
                    "lf_content",
                    "tail_robust",
                    "attention_geometry",
                )
            }
        ),
        "branch_risk_records": branch_risk_records,
        "branch_risk_content_digest": build_stable_digest(
            {
                "semantic_routing_enabled": True,
                "active_carrier_branches": [
                    "lf_content",
                    "tail_robust",
                    "attention_geometry",
                ],
                "risk_signal_content_records": {
                    "current_decoded_rgb_content_sha256": "6" * 64,
                    "previous_step_decoded_rgb_content_sha256": "7" * 64,
                    "clip_patch_tokens_content_sha256": "8" * 64,
                    "clip_cls_token_content_sha256": "9" * 64,
                    "semantic_risk_signal_content_sha256": "1" * 64,
                    "texture_risk_signal_content_sha256": "2" * 64,
                    "local_contrast_risk_signal_content_sha256": "3" * 64,
                    "adjacent_step_stability_signal_content_sha256": "4" * 64,
                    "attention_stability_signal_content_sha256": "5" * 64,
                },
                "branch_risk_content_records": branch_risk_content_records,
            }
        ),
        "null_space_records": null_space_records,
        "lf_template_content_sha256": lf_template_identity[
            "canonical_template_content_sha256"
        ],
        "lf_template_digest": lf_template_identity["template_digest"],
        "lf_template_shape": lf_template_identity["template_shape"],
        "lf_projection_energy_retention": 0.2,
        "tail_template_digest": tail_template_identity["template_digest"],
        "tail_template_content_sha256": tail_template_identity[
            "canonical_template_content_sha256"
        ],
        "tail_template_shape": tail_template_identity["template_shape"],
        "tail_template_element_count": math.prod(
            tail_template_identity["template_shape"]
        ),
        "tail_selected_element_count": math.ceil(
            math.prod(tail_template_identity["template_shape"])
            * config.tail_fraction
        ),
        "tail_threshold": 1.0,
        "tail_retained_fraction": (
            math.ceil(
                math.prod(tail_template_identity["template_shape"])
                * config.tail_fraction
            )
            / math.prod(tail_template_identity["template_shape"])
        ),
        "tail_projection_energy_retention": 0.2,
        "attention_score_before": 0.10,
        "attention_content_base_score": 0.11,
        "attention_score_after": 0.13,
        "attention_actual_written_content_base_score": 0.105,
        "attention_final_combined_score": 0.12,
        "attention_score_gain": 0.02,
        "attention_applied_update_strength": 0.005,
        "attention_update_content_sha256": "7" * 64,
        "attention_update_unit_direction_content_sha256": "d" * 64,
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
        "attention_relation_qk_operator_metadata_records": (
            qk_operator_metadata_records
        ),
        "attention_relation_qk_operator_metadata_digest": (
            qk_operator_metadata_records_digest(
                qk_operator_metadata_records
            )
        ),
        "attention_relation_qk_operator_metadata_ready": True,
        "attention_module_names": list(config.attention_module_names),
        "attention_coordinate_convention": (
            config.attention_coordinate_convention
        ),
        "attention_grid_align_corners": (
            config.attention_grid_align_corners
        ),
        "quantized_write_update_content_sha256": "d" * 64,
        "quantized_write_update_dtype": "torch.float16",
        "quantized_write_update_shape": [1, 16, 64, 64],
        "quantized_write_update_norm": 0.01,
        "quantized_composition_evidence_version": (
            QUANTIZED_COMPOSITION_EVIDENCE_VERSION
        ),
        "quantized_write_original_latent_content_sha256": "a" * 64,
        "quantized_write_candidate_latent_content_sha256": "b" * 64,
        "combined_update_content_sha256": "e" * 64,
        "combined_budget_envelope_content_sha256": "f" * 64,
        "quantized_write_composition_order": list(
            config.quantized_branch_composition_order
        ),
        "quantized_write_common_scale": 0.5,
        "quantized_write_backtracking_factor": (
            config.quantized_budget_envelope_backtracking_factor
        ),
        "quantized_write_backtracking_step_count": 1,
        "quantized_write_active_branch_order": [
            "lf_content",
            "tail_robust",
            "attention_geometry",
        ],
        "quantized_write_branch_content_identities": {
            name: {
                "branch_written_update_content_sha256": (
                    branch_risk_records[name][
                        "branch_written_update_content_sha256"
                    ]
                ),
                "branch_budget_envelope_content_sha256": (
                    branch_risk_records[name][
                        "branch_budget_envelope_content_sha256"
                    ]
                ),
            }
            for name in (
                "lf_content",
                "tail_robust",
                "attention_geometry",
            )
        },
        "quantized_write_maximum_envelope_ratio": 1.0,
        "quantized_write_budget_envelope_ready": True,
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
        "quantized_write_reference_feature_content_sha256": "1" * 64,
        "quantized_write_jacobian_response_content_sha256": "2" * 64,
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
        "metadata": {
            "semantic_routing_enabled": True,
            "null_space_enabled": True,
            "lf_enabled": True,
            "tail_robust_enabled": True,
            "tail_truncation_enabled": True,
            "attention_geometry_enabled": True,
        },
    }
    record["quantized_composition_evidence_digest"] = (
        recompute_quantized_composition_evidence_digest(record)
    )
    assert _scientific_update_record_ready(record, config) is True
    record["lf_carrier_protocol_digest"] = "0" * 64
    assert _scientific_update_record_ready(record, config) is False
    record["lf_carrier_protocol_digest"] = (
        _FORMAL_LOW_FREQUENCY_CONFIG.protocol_digest
    )
    record["lf_template_digest"] = ""
    assert _scientific_update_record_ready(record, config) is False
    record["lf_template_digest"] = lf_template_identity["template_digest"]
    removed_signal = record.pop("semantic_risk_signal_content_sha256")
    assert _scientific_update_record_ready(record, config) is False
    record["semantic_risk_signal_content_sha256"] = removed_signal
    removed_envelope = record.pop(
        "combined_budget_envelope_content_sha256"
    )
    assert _scientific_update_record_ready(record, config) is False
    record["combined_budget_envelope_content_sha256"] = removed_envelope
    record["quantized_write_common_scale"] = 0.25
    assert _scientific_update_record_ready(record, config) is False
    record["quantized_write_common_scale"] = 0.5
    record["quantized_write_update_dtype"] = "torch.float32"
    assert _scientific_update_record_ready(record, config) is False
    record["quantized_write_update_dtype"] = "torch.float16"
    record["quantized_composition_evidence_digest"] = "0" * 64
    assert _scientific_update_record_ready(record, config) is False
    record["quantized_composition_evidence_digest"] = (
        recompute_quantized_composition_evidence_digest(record)
    )
    record["quantized_write_update_norm"] = 0.0
    assert _scientific_update_record_ready(record, config) is False
    record["quantized_write_update_norm"] = 0.01
    record["quantized_write_jacobian_response_norm"] = float("nan")
    assert _scientific_update_record_ready(record, config) is False
    record["quantized_write_jacobian_response_norm"] = 1e-5
    record["quantized_write_reference_feature_norm"] = -1.0
    assert _scientific_update_record_ready(record, config) is False
    record["quantized_write_reference_feature_norm"] = 1.0
    reference_feature_content_sha256 = record.pop(
        "quantized_write_reference_feature_content_sha256"
    )
    assert _scientific_update_record_ready(record, config) is False
    record["quantized_write_reference_feature_content_sha256"] = (
        reference_feature_content_sha256
    )
    response_content_sha256 = record.pop(
        "quantized_write_jacobian_response_content_sha256"
    )
    assert _scientific_update_record_ready(record, config) is False
    record["quantized_write_jacobian_response_content_sha256"] = (
        response_content_sha256
    )
    record["branch_risk_records"]["lf_content"].pop(
        "branch_written_update_content_sha256"
    )
    assert _scientific_update_record_ready(record, config) is False
    record["branch_risk_records"]["lf_content"][
        "branch_written_update_content_sha256"
    ] = branch_update_content_records["lf_content"]
    record["branch_risk_records"]["lf_content"][
        "branch_post_risk_relative_response_residual"
    ] = 1e-3
    assert _scientific_update_record_ready(record, config) is False
    record["branch_risk_records"]["lf_content"][
        "branch_post_risk_relative_response_residual"
    ] = 1e-5
    record["branch_risk_records"]["lf_content"][
        "branch_written_update_maximum_envelope_ratio"
    ] = -1.0
    assert _scientific_update_record_ready(record, config) is False
    record["branch_risk_records"]["lf_content"][
        "branch_written_update_maximum_envelope_ratio"
    ] = 1.0
    record["attention_score_gain"] = 0.0
    assert _scientific_update_record_ready(record, config) is False
    record["attention_score_gain"] = 0.02
    record["attention_score_after"] = 0.11
    assert _scientific_update_record_ready(record, config) is False
    record["attention_score_after"] = 0.13
    record["attention_actual_written_content_base_score"] = 0.12
    assert _scientific_update_record_ready(record, config) is False
    record["attention_actual_written_content_base_score"] = 0.105
    record["branch_risk_bundle_digest"] = "0" * 64
    assert _scientific_update_record_ready(record, config) is False
    record["branch_risk_bundle_digest"] = build_stable_digest(
        {
            name: branch_risk_records[name]["risk_field_digest"]
            for name in (
                "lf_content",
                "tail_robust",
                "attention_geometry",
            )
        }
    )
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
    record["null_space_records"]["lf_content"][
        "column_response_norms"
    ][0] = 0.25
    assert _scientific_update_record_ready(record, config) is False
    record["null_space_records"]["lf_content"][
        "column_response_norms"
    ][0] = 1e-6
    record["attention_relation_qk_operator_metadata_records"][0][
        "head_count"
    ] = 3
    assert _scientific_update_record_ready(record, config) is False
    record["attention_relation_qk_operator_metadata_records"][0][
        "head_count"
    ] = 2
    record["attention_relation_qk_operator_metadata_digest"] = "0" * 64
    assert _scientific_update_record_ready(record, config) is False
    record["attention_relation_qk_operator_metadata_digest"] = (
        qk_operator_metadata_records_digest(
            record[
                "attention_relation_qk_operator_metadata_records"
            ]
        )
    )
    record["attention_qk_atomic_content_records"][0][
        "evaluation_score"
    ] = 0.20
    record["attention_qk_atomic_content_digest"] = (
        qk_atomic_evaluation_records_digest(
            record["attention_qk_atomic_content_records"],
            "attention_qk_atomic_content_records",
        )
    )
    assert _scientific_update_record_ready(record, config) is False
    record["attention_qk_atomic_content_records"][0][
        "evaluation_score"
    ] = 0.10
    record["attention_qk_atomic_content_digest"] = (
        qk_atomic_evaluation_records_digest(
            record["attention_qk_atomic_content_records"],
            "attention_qk_atomic_content_records",
        )
    )
    second_qk_record = record["attention_qk_atomic_content_records"][1]
    second_atoms = second_qk_record["qk_atomic_content_records"]
    second_digest = second_qk_record["qk_atomic_content_digest"]
    second_qk_record["qk_atomic_content_records"] = record[
        "attention_qk_atomic_content_records"
    ][0]["qk_atomic_content_records"]
    second_qk_record["qk_atomic_content_digest"] = record[
        "attention_qk_atomic_content_records"
    ][0]["qk_atomic_content_digest"]
    record["attention_qk_atomic_content_digest"] = (
        qk_atomic_evaluation_records_digest(
            record["attention_qk_atomic_content_records"],
            "attention_qk_atomic_content_records",
        )
    )
    assert _scientific_update_record_ready(record, config) is False
    second_qk_record["qk_atomic_content_records"] = second_atoms
    second_qk_record["qk_atomic_content_digest"] = second_digest
    record["attention_qk_atomic_content_digest"] = (
        qk_atomic_evaluation_records_digest(
            record["attention_qk_atomic_content_records"],
            "attention_qk_atomic_content_records",
        )
    )
    record["attention_module_names"] = ["transformer_blocks.1.attn"]
    assert _scientific_update_record_ready(record, config) is False


@pytest.mark.quick
def test_tail_robust_template_records_amplitude_tail_semantics() -> None:
    """尾部截断应改变稀疏率, 并记录幅值尾部语义。"""

    latent = torch.zeros(1, 2, 8, 8)
    lf_template = build_low_frequency_template(
        latent,
        "key",
        "model",
        _FORMAL_LOW_FREQUENCY_CONFIG,
        prg_version=KEYED_PRG_VERSION,
    )
    tail_template, _, retained_fraction = build_tail_robust_template(
        latent,
        "key",
        "model",
        0.20,
        prg_version=KEYED_PRG_VERSION,
    )
    observed = 0.7 * lf_template + 0.3 * tail_template
    score = compute_blind_content_score(
        observed,
        lf_template,
        tail_template,
        _FORMAL_LF_WEIGHT,
        _FORMAL_TAIL_ROBUST_WEIGHT,
    )

    assert 0.15 <= retained_fraction <= 0.25
    assert int(torch.count_nonzero(tail_template).item()) == 26
    assert score.content_score > 0.5
    assert score.metadata["tail_branch_semantics"] == "gaussian_amplitude_tail_truncation"


@pytest.mark.quick
def test_tail_truncation_ablation_skips_amplitude_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关闭尾部截断时应保留全部高斯元素, 且不得执行幅值排序."""

    import main.methods.carrier.keyed_tensor as keyed_tensor_module

    def fail_ranking(*_args: object, **_kwargs: object) -> object:
        """幅值排序一旦被调用就使消融反例测试失败."""

        pytest.fail("tail_fraction=1.0 时仍执行幅值排序")

    monkeypatch.setattr(
        keyed_tensor_module,
        "sorted",
        fail_ranking,
        raising=False,
    )
    template, threshold, retained_fraction = build_tail_robust_template(
        torch.zeros(1, 2, 8, 8),
        "tail_no_truncation_key",
        "tail_no_truncation_model",
        1.0,
        prg_version=KEYED_PRG_VERSION,
    )
    protocol = tail_robust_carrier_protocol_record(
        1.0,
        prg_version=KEYED_PRG_VERSION,
    )

    assert threshold == 0.0
    assert retained_fraction == 1.0
    assert int(torch.count_nonzero(template).item()) == template.numel()
    assert float(torch.linalg.norm(template).item()) == pytest.approx(1.0)
    assert protocol["tail_selection_rule"] == (
        "all_elements_without_amplitude_ranking"
    )


@pytest.mark.quick
def test_keyed_templates_use_versioned_device_independent_prg() -> None:
    """密钥模板必须由固定 PRG 算法生成, 设备 RNG 不得参与定义."""

    reference = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    first_lf = build_low_frequency_template(
        reference,
        "known-key",
        "known-model",
        _FORMAL_LOW_FREQUENCY_CONFIG,
        prg_version=KEYED_PRG_VERSION,
    )
    second_lf = build_low_frequency_template(
        reference,
        "known-key",
        "known-model",
        _FORMAL_LOW_FREQUENCY_CONFIG,
        prg_version=KEYED_PRG_VERSION,
    )
    tail, threshold, retained_fraction = build_tail_robust_template(
        reference,
        "known-key",
        "known-model",
        0.25,
        prg_version=KEYED_PRG_VERSION,
    )
    protocol = keyed_prg_protocol_record()
    uniform_vector = keyed_prg_module.build_keyed_uniform_tensor(
        (4,),
        "known-key",
        {"operator": "known_answer_uniform", "role": "cpu_test"},
    )
    gaussian_vector = keyed_prg_module.build_keyed_gaussian_tensor(
        (4,),
        "known-key",
        {"operator": "known_answer_gaussian", "role": "cpu_test"},
    )

    assert KEYED_PRG_VERSION == (
        "sha256_counter_normal_icdf_table20_float32_v2"
    )
    assert torch.equal(first_lf, second_lf)
    assert hashlib.sha256(
        first_lf.detach().contiguous().numpy().tobytes()
    ).hexdigest() == "2c265021651407651263b6253a6ee7dbbb540bd879f248b583963cf401629519"
    assert hashlib.sha256(
        tail.detach().contiguous().numpy().tobytes()
    ).hexdigest() == "9bd7ac53fec2166cd4c82accf3ed4bef1a60c3114d4f71ae2edcd95f408bdc2d"
    assert threshold == pytest.approx(1.0803799629211426)
    assert retained_fraction == 0.25
    assert protocol["canonical_generation_device"] == "cpu"
    assert protocol["counter_initial_value"] == 0
    assert protocol["counter_bytes"] == 16
    assert protocol["word_offsets"] == [0, 8, 16, 24]
    assert protocol["uniform_mapping"] == "(mantissa+1)/(2^53+2)"
    assert protocol["normal_index_bits"] == 20
    assert protocol["normal_bitstream_order"] == (
        "sha256_blocks_then_msb_first_bits"
    )
    assert protocol["normal_quantile_table_sha256"] == (
        "70abf440a7f3670147965ffa52f5aaa639dab97f6282b68f3a9a1b1ce5e6cf5a"
    )
    assert uniform_vector.tolist() == pytest.approx(
        (
            0.15819059312343597,
            0.2266322374343872,
            0.11157729476690292,
            0.9893344044685364,
        )
    )
    assert gaussian_vector.tolist() == pytest.approx(
        (
            0.9188238978385925,
            0.9646357297897339,
            1.737541675567627,
            0.24967548251152039,
        )
    )
    assert protocol["keyed_prg_protocol_digest"] == (
        "a6266dc1fb4a59f8038062dcd120f145582153138b8176baae12013d5a22687b"
    )
    assert tensor_content_sha256(uniform_vector) == (
        "226041062e606ceebc26f75cda2479c32e216890495a67ef5d1ad37f76f084e5"
    )
    assert tensor_content_sha256(gaussian_vector) == (
        "2e0775d43b078333c58079eeb0b595bfd2a7491bc09ab430b5da69f323edd0dc"
    )

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
        KEYED_PRG_VERSION,
    )
    assert torch.equal(first_candidates, second_candidates)
    assert torch.equal(relation_signs, relation_signs.transpose(0, 1))
    assert torch.count_nonzero(torch.diag(relation_signs)).item() == 0
    assert relation_signs.tolist() == [
        [0.0, -1.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0, -1.0],
        [1.0, 1.0, 0.0, -1.0],
        [1.0, -1.0, -1.0, 0.0],
    ]

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
            _FORMAL_LOW_FREQUENCY_CONFIG,
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
    assert ATTENTION_IMAGE_RESAMPLING_MODE == "bilinear"
    assert ATTENTION_IMAGE_PADDING_MODE == "border"
    assert ATTENTION_IMAGE_QUANTIZATION_PROTOCOL == (
        "clamp_0_1_multiply_255_floor_uint8_rgb_v1"
    )
    assert [aligned.getpixel((column, 1))[0] for column in range(3)] == [
        100,
        200,
        200,
    ]


@pytest.mark.quick
def test_image_alignment_quantizes_fractional_rgb_with_floor() -> None:
    """对齐后的连续 RGB 值必须按冻结协议向下量化而非四舍五入."""

    image = Image.new("RGB", (2, 2))
    image.putdata([(0, 0, 0), (4, 4, 4), (0, 0, 0), (4, 4, 4)])
    alignment = SimpleNamespace(
        affine_transform=((1.0, 0.0, 0.28), (0.0, 1.0, 0.0))
    )

    aligned = _align_image(image, alignment)

    assert aligned.getpixel((0, 0)) == (0, 0, 0)
    assert aligned.getpixel((1, 0)) == (4, 4, 4)


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
        prg_version=KEYED_PRG_VERSION,
    )
    assert identity.qk_operator_metadata_ready is True
    assert len(identity.qk_operator_metadata_digest) == 64
    assert identity.qk_atomic_content_ready is True
    assert len(identity.qk_atomic_content_records) == 1
    assert len(identity.qk_atomic_content_digest) == 64
    assert identity.component_identity_digest == (
        "d69e5a0605182aae07f610258557ad5fb6913d0c7d526ae841f41c86fe31477f"
    )
    assert identity.keyed_projection_digest == (
        "837034707b9ffb733c123b629aa0a013ce4416ee6b5a212b7060ab32bbdcde7c"
    )


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
@pytest.mark.parametrize("invalid_heads", [None, True, 0, -1, 1.5])
def test_qk_relation_requires_explicit_positive_integer_head_count(
    invalid_heads: object,
) -> None:
    """核心 Q/K 算子必须拒绝缺失或非法 heads, 不得静默退化成单头。"""

    module = _ToyAttention(4)
    if invalid_heads is None:
        del module.heads
    else:
        module.heads = invalid_heads

    with pytest.raises(TypeError, match="正整数 heads"):
        qk_self_attention(
            module,
            torch.randn(1, 4, 4),
            max_tokens=4,
            layer_name="invalid_head_count_layer",
        )


@pytest.mark.quick
def test_attention_recorder_rejects_missing_hidden_state_tensor() -> None:
    """冻结层钩子必须立即拒绝无 Tensor 输入, 不得静默漏记 Q/K 原子。"""

    module = _ToyAttention(4)
    with DifferentiableAttentionRecorder(
        (("required_attention_layer", module),),
        max_tokens=4,
    ):
        with pytest.raises(RuntimeError, match="没有提供可核验"):
            module("not-a-tensor")  # type: ignore[arg-type]


@pytest.mark.quick
def test_attention_stability_comes_from_multiple_real_qk_layers() -> None:
    """相同 Q/K 关系层应产生接近 1 的真实关系稳定图。"""

    logits = torch.randn(1, 4, 4)
    records = (
        (
            "layer_a",
            _direct_qk_relation_from_logits(logits, "layer_a"),
            (0, 1, 2, 3),
        ),
        (
            "layer_b",
            _direct_qk_relation_from_logits(logits.clone(), "layer_b"),
            (0, 1, 2, 3),
        ),
    )

    stability = attention_relation_stability_map(records, (4, 4))

    assert stability.shape == (1, 4, 4)
    assert float(stability.min()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.quick
@pytest.mark.parametrize(
    "invalid_relation",
    ["bare_tensor", "wrong_source", "missing_metadata"],
)
def test_attention_stability_and_selection_require_direct_qk_identity(
    invalid_relation: str,
) -> None:
    """稳定度与 token 选择必须拒绝裸概率或错误来源关系。"""

    logits = torch.randn(1, 4, 4)
    if invalid_relation == "bare_tensor":
        relation: object = torch.softmax(logits, dim=-1)
    elif invalid_relation == "wrong_source":
        direct = _direct_qk_relation_from_logits(logits, "layer_a")
        relation = QKAttentionRelation(
            centered_logits=direct.centered_logits,
            probabilities=direct.probabilities,
            relation_source="untrusted_attention_output",
            metadata=dict(direct.metadata),
        )
    else:
        relation = QKAttentionRelation(
            centered_logits=logits - logits.mean(dim=-1, keepdim=True),
            probabilities=torch.softmax(logits, dim=-1),
        )
    second_relation = (
        _direct_qk_relation_from_logits(logits.clone(), "layer_b")
        if invalid_relation == "bare_tensor"
        else relation.clone() if hasattr(relation, "clone") else relation
    )
    records = (
        ("layer_a", relation, (0, 1, 2, 3)),
        ("layer_b", second_relation, (0, 1, 2, 3)),
    )

    with pytest.raises(ValueError, match="直接 Q/K"):
        attention_relation_stability_map(records, (4, 4))
    with pytest.raises(ValueError, match="直接 Q/K"):
        select_stable_attention_tokens(records, stable_token_fraction=0.5)


@pytest.mark.quick
@pytest.mark.parametrize("mismatch", ["layer_name", "token_indices"])
def test_attention_records_bind_outer_layer_and_token_identity(
    mismatch: str,
) -> None:
    """外层 Q/K 记录不得改名或更换 token 索引后冒充另一科学原子。"""

    logits = torch.randn(1, 4, 4)
    layer_a = _direct_qk_relation_from_logits(logits, "layer_a")
    layer_b = _direct_qk_relation_from_logits(logits.clone(), "layer_b")
    first_layer_name = "forged_layer" if mismatch == "layer_name" else "layer_a"
    first_token_indices = (
        (0, 3, 12, 15)
        if mismatch == "token_indices"
        else (0, 1, 2, 3)
    )
    records = (
        (first_layer_name, layer_a, first_token_indices),
        ("layer_b", layer_b, (0, 1, 2, 3)),
    )

    with pytest.raises(ValueError, match="内部身份不一致"):
        attention_relation_stability_map(records, (4, 4))
    with pytest.raises(ValueError, match="内部身份不一致"):
        select_stable_attention_tokens(records, stable_token_fraction=0.5)


@pytest.mark.quick
def test_attention_stability_rejects_duplicate_layer_identity() -> None:
    """同一 Q/K 层的克隆不得通过改写记录数量冒充跨层稳定性。"""

    logits = torch.randn(1, 4, 4)
    first = _direct_qk_relation_from_logits(logits, "layer_a")
    second = _direct_qk_relation_from_logits(logits.clone(), "layer_a")
    records = (
        ("layer_a", first, (0, 1, 2, 3)),
        ("layer_a", second, (0, 1, 2, 3)),
    )

    with pytest.raises(ValueError, match="不得用同一层"):
        attention_relation_stability_map(records, (4, 4))
    with pytest.raises(ValueError, match="不得用同一层"):
        select_stable_attention_tokens(records, stable_token_fraction=0.5)


@pytest.mark.quick
def test_stable_attention_tokens_drive_keyed_geometry_score() -> None:
    """稳定 token 集必须真实改变 Q/K 目标权重并保存可复现身份。"""

    generator = torch.Generator().manual_seed(1703)
    logits = torch.randn(1, 9, 9, generator=generator)
    token_indices = tuple(range(9))
    records = (
        (
            "layer_a",
            _direct_qk_relation_from_logits(logits, "layer_a"),
            token_indices,
        ),
        (
            "layer_b",
            _direct_qk_relation_from_logits(logits.clone(), "layer_b"),
            token_indices,
        ),
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
        prg_version=KEYED_PRG_VERSION,
        stable_pair_weights=pair_weights,
    )
    full = attention_geometry_score(
        records,
        "stable_token_key",
        prg_version=KEYED_PRG_VERSION,
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
        KEYED_PRG_VERSION,
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
def test_attention_row_correlation_uses_independent_energy_thresholds() -> None:
    """两个能量必须分别过阈值, 不得以能量乘积替代独立有效性判断。"""

    generator = torch.Generator().manual_seed(260715)
    base = torch.randn(1, 4, 4, 4, generator=generator)
    pair_weights = 1.0 - torch.eye(4)

    jointly_small = base * 1e-8
    jointly_small_scores = attention_relation_component_scores(
        jointly_small,
        jointly_small,
        pair_weights,
    )
    assert torch.allclose(
        jointly_small_scores,
        torch.ones_like(jointly_small_scores),
        atol=1e-5,
        rtol=0.0,
    )

    one_sided_degenerate = base.clone()
    one_sided_projection = base.clone()
    one_sided_degenerate[..., 0] *= 1e-13
    one_sided_projection[..., 0] *= 1e13
    one_sided_degenerate.requires_grad_(True)
    one_sided_projection.requires_grad_(True)
    one_sided_scores = attention_relation_component_scores(
        one_sided_degenerate,
        one_sided_projection,
        pair_weights,
    )
    assert float(one_sided_scores[0, 0]) == 0.0
    one_sided_scores.sum().backward()
    assert bool(torch.isfinite(one_sided_degenerate.grad).all())
    assert bool(torch.isfinite(one_sided_projection.grad).all())


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
def test_leave_one_component_out_short_circuits_descriptor_projection_and_score() -> None:
    """留一消融必须清空被禁用通道, 且异常原子不得进入相关分数."""

    generator = torch.Generator().manual_seed(260716)
    relation = _direct_qk_relation_from_logits(
        torch.randn(1, 9, 9, generator=generator)
    )
    token_indices = tuple(range(9))
    complete_descriptor = build_attention_relation_descriptor(
        relation,
        token_indices,
    )
    complete_projection = keyed_attention_relation_projection(
        complete_descriptor,
        "leave_one_short_circuit_key",
        "leave_one_short_circuit_layer",
        KEYED_PRG_VERSION,
    )
    pair_weights = 1.0 - torch.eye(9)

    for removed_index, removed_name in enumerate(
        ATTENTION_RELATION_COMPONENT_NAMES
    ):
        weights = tuple(
            0.0 if index == removed_index else 1.0 / 3.0
            for index in range(len(ATTENTION_RELATION_COMPONENT_NAMES))
        )
        descriptor = build_attention_relation_descriptor(
            relation,
            token_indices,
            weights,
        )
        projection = keyed_attention_relation_projection(
            descriptor,
            "leave_one_short_circuit_key",
            "leave_one_short_circuit_layer",
            KEYED_PRG_VERSION,
            weights,
        )

        assert removed_name not in descriptor.active_component_names
        assert torch.count_nonzero(
            descriptor.values[..., removed_index]
        ).item() == 0
        assert torch.count_nonzero(
            projection.values[..., removed_index]
        ).item() == 0
        for active_index in range(len(ATTENTION_RELATION_COMPONENT_NAMES)):
            if active_index == removed_index:
                continue
            assert torch.allclose(
                descriptor.values[..., active_index],
                complete_descriptor.values[..., active_index],
            )
            assert torch.equal(
                projection.values[..., active_index],
                complete_projection.values[..., active_index],
            )

        poisoned_relation = descriptor.values.clone()
        poisoned_projection = projection.values.clone()
        poisoned_relation[..., removed_index] = torch.nan
        poisoned_projection[..., removed_index] = torch.nan
        component_scores = attention_relation_component_scores(
            poisoned_relation,
            poisoned_projection,
            pair_weights,
            component_weights=weights,
        )
        component_scores[..., removed_index] = torch.nan
        total_score = combine_attention_relation_component_scores(
            component_scores,
            weights,
        )

        assert torch.isfinite(total_score).all()
        assert torch.isfinite(
            component_scores[..., [
                index
                for index in range(len(ATTENTION_RELATION_COMPONENT_NAMES))
                if index != removed_index
            ]]
        ).all()


@pytest.mark.quick
@pytest.mark.parametrize("removed_index", (1, 3))
def test_expensive_attention_component_operator_is_not_called_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    removed_index: int,
) -> None:
    """禁用 soft-rank 或距离调制时不得执行对应科学算子."""

    import main.methods.geometry.differentiable_attention as attention_module

    relation = _direct_qk_relation_from_logits(
        torch.randn(1, 9, 9, generator=torch.Generator().manual_seed(260717))
    )
    weights = tuple(
        0.0 if index == removed_index else 1.0 / 3.0
        for index in range(len(ATTENTION_RELATION_COMPONENT_NAMES))
    )

    def fail_operator(*_args: object, **_kwargs: object) -> object:
        """被禁用的算子一旦执行就立即使测试失败."""

        pytest.fail("已禁用 attention 分量仍执行其专属算子")

    if removed_index == 1:
        monkeypatch.setattr(
            attention_module,
            "_differentiable_row_rank",
            fail_operator,
        )
    else:
        monkeypatch.setattr(torch, "cdist", fail_operator)

    identity = build_attention_relation_graph_identity(
        (("test_qk_relation_layer", relation, tuple(range(9))),),
        "disabled_attention_operator_key",
        prg_version=KEYED_PRG_VERSION,
        component_weights=weights,
    )

    assert ATTENTION_RELATION_COMPONENT_NAMES[removed_index] not in (
        identity.active_component_names
    )


@pytest.mark.quick
def test_differentiable_soft_rank_contributes_nonzero_logit_gradient() -> None:
    """soft-rank 分量必须对真实 Q/K logits 保留非零可微梯度。"""

    generator = torch.Generator().manual_seed(260713)
    logits = torch.randn(1, 9, 9, generator=generator).requires_grad_(True)
    relation = _direct_qk_relation_from_logits(logits)
    descriptor = build_attention_relation_descriptor(relation, tuple(range(9)))
    projection = keyed_attention_relation_projection(
        descriptor,
        "soft_rank_gradient_key",
        "soft_rank_gradient_layer",
        KEYED_PRG_VERSION,
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
    relation = _direct_qk_relation_from_logits(logits)
    descriptor = build_attention_relation_descriptor(relation, tuple(range(9)))
    projection = keyed_attention_relation_projection(
        descriptor,
        "distance_modulation_key",
        "distance_modulation_layer",
        KEYED_PRG_VERSION,
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
        routed_candidate_matrix=identity,
        routed_candidate_response_matrix=response_matrix,
        projected_direction_matrix=identity,
        projected_direction_response_matrix=response_matrix,
        latent_basis=identity,
        basis_response_matrix=response_matrix,
        basis_reference_matrix=identity,
        basis_reference_response_matrix=response_matrix,
        column_response_norms=(0.0,) * element_count,
        column_reference_response_norms=(0.0,) * element_count,
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
        routed_candidate_response_matrix_content_sha256=tensor_content_sha256(
            response_matrix
        ),
        projected_direction_matrix_content_sha256=tensor_content_sha256(identity),
        projected_direction_response_matrix_content_sha256=tensor_content_sha256(
            response_matrix
        ),
        latent_basis_content_sha256=tensor_content_sha256(identity),
        basis_response_matrix_content_sha256=tensor_content_sha256(response_matrix),
        basis_reference_response_matrix_content_sha256=tensor_content_sha256(
            response_matrix
        ),
        solver_digest="identity_test_basis",
        metadata={},
    )


@pytest.mark.quick
def test_null_space_projection_keeps_float32_for_float16_input() -> None:
    """安全投影不得在风险包络和 JVP 复验前恢复为低精度 dtype。"""

    latent = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float16)
    projected = _identity_null_space(latent).project(latent)

    assert projected.dtype == torch.float32
    assert torch.equal(projected, latent.float())


def _attention_risk_bound_fixture(
    direction: torch.Tensor,
    strength: float,
) -> RiskBoundedUpdate:
    """从真实内容基底 Q/K 梯度构造测试用风险有界对象。"""

    unit_direction = direction.detach().float()
    unit_direction = unit_direction / unit_direction.norm()
    scalar = torch.tensor([strength], dtype=torch.float32)
    update = unit_direction * strength
    envelope = torch.full_like(update, update.abs().max())
    return RiskBoundedUpdate(
        branch_name="attention_geometry",
        unit_direction=unit_direction,
        effective_budget=torch.ones_like(update),
        amplitude_envelope=envelope,
        update=update,
        nominal_strength=scalar.clone(),
        applied_strength=scalar,
        risk_scale_factor=torch.ones_like(scalar),
        maximum_envelope_ratio=torch.ones_like(scalar),
        budget_ceiling=1.0,
        direction_epsilon=1e-12,
        numerical_epsilon=1e-12,
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
        original_gradient = compute_attention_geometry_gradient(
            latent,
            module,
            recorder,
            "attention_key",
            prg_version=KEYED_PRG_VERSION,
        )
        update = optimize_attention_geometry_update(
            latent=latent,
            transformer_forward=module,
            recorder=recorder,
            key_material="attention_key",
            safe_subspace=_identity_null_space(latent),
            risk_bounded_update=_attention_risk_bound_fixture(
                original_gradient.gradient,
                0.05,
            ),
            precomputed_gradient=original_gradient,
            precomputed_content_base_gradient=original_gradient,
            prg_version=KEYED_PRG_VERSION,
        )

    assert original_gradient.evaluation_latent_content_sha256 == (
        attention_module.tensor_content_sha256(latent.detach().float())
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
        "optimization_content_base_latent",
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
        original_gradient = compute_attention_geometry_gradient(
            latent,
            module,
            recorder,
            "combined_attention_key",
            prg_version=KEYED_PRG_VERSION,
        )
        _, content_base_latent, _ = compose_ordered_float32_update_once(
            original_latent=latent,
            branch_update_tensors={"lf_content": content_base_update.float()},
            common_scale=1.0,
        )
        content_base_gradient = compute_attention_geometry_gradient(
            content_base_latent,
            module,
            recorder,
            "combined_attention_key",
            prg_version=KEYED_PRG_VERSION,
            stable_token_selection=StableAttentionTokenSelection(
                token_positions=original_gradient.stable_token_positions,
                token_indices=original_gradient.stable_token_indices,
                stable_token_fraction=original_gradient.stable_token_fraction,
                selection_digest=(
                    original_gradient.stable_token_selection_digest
                ),
            ),
        )
        update = optimize_attention_geometry_update(
            latent=latent,
            transformer_forward=module,
            recorder=recorder,
            key_material="combined_attention_key",
            safe_subspace=_identity_null_space(latent),
            risk_bounded_update=_attention_risk_bound_fixture(
                content_base_gradient.gradient,
                0.05,
            ),
            precomputed_gradient=original_gradient,
            precomputed_content_base_gradient=content_base_gradient,
            prg_version=KEYED_PRG_VERSION,
            base_update=content_base_update,
        )
        recorder.clear()
        _, actual_candidate, _ = compose_ordered_float32_update_once(
            original_latent=latent,
            branch_update_tensors={
                "lf_content": content_base_update.float(),
                "attention_geometry": update.update.float(),
            },
            common_scale=1.0,
        )
        module(actual_candidate)
        actual_score = float(
            attention_geometry_score(
                recorder.records,
                "combined_attention_key",
                prg_version=KEYED_PRG_VERSION,
            ).detach().item()
        )

    assert content_base_gradient.evaluation_latent_content_sha256 == (
        attention_module.tensor_content_sha256(
            content_base_latent.detach().float()
        )
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
        KEYED_PRG_VERSION,
    )
    canonical_logits = (2.0 * relation_signs).unsqueeze(0)
    index = torch.tensor(permutation, dtype=torch.long)
    observed = _direct_qk_relation_from_logits(
        canonical_logits.index_select(1, index).index_select(2, index),
        layer_name,
    )
    replicate_layer_name = f"{layer_name}_replicate"
    observed_replicate = _direct_qk_relation_from_logits(
        observed.centered_logits.clone(),
        replicate_layer_name,
    )

    result = recover_attention_affine_alignment(
        observed,
        key_material,
        layer_name,
        tuple(range(token_count)),
        build_stable_attention_pair_weights(
                (
                    (layer_name, observed, tuple(range(token_count))),
                    (
                        replicate_layer_name,
                        observed_replicate,
                        tuple(range(token_count)),
                    ),
                ),
                select_stable_attention_tokens(
                    (
                        (layer_name, observed, tuple(range(token_count))),
                        (
                            replicate_layer_name,
                            observed_replicate,
                            tuple(range(token_count)),
                        ),
                    )
                ),
        ),
        prg_version=KEYED_PRG_VERSION,
        anchor_count=12,
        residual_threshold=0.20,
        minimum_inlier_ratio=0.50,
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
    layer_name, second_layer_name = FROZEN_SD35_ATTENTION_MODULE_NAMES
    relation_signs = keyed_relation_signs(
        torch.zeros(1, token_count, token_count),
        key_material,
        layer_name,
        KEYED_PRG_VERSION,
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
    second_relation_signs = keyed_relation_signs(
        torch.zeros(1, token_count, token_count),
        key_material,
        second_layer_name,
        KEYED_PRG_VERSION,
    )
    second_canonical_logits = (2.0 * second_relation_signs).unsqueeze(0)
    second_canonical_attention = _direct_qk_relation_from_logits(
        second_canonical_logits,
        second_layer_name,
    )
    second_observed_attention = _direct_qk_relation_from_logits(
        second_canonical_logits.index_select(1, flip).index_select(2, flip),
        second_layer_name,
    )
    reference = torch.zeros(1, 2, 8, 8)
    lf_template = build_low_frequency_template(
        reference,
        key_material,
        model_id,
        _FORMAL_LOW_FREQUENCY_CONFIG,
        prg_version=KEYED_PRG_VERSION,
    )
    tail_template = build_tail_robust_template(
        reference,
        key_material,
        model_id,
        0.20,
        prg_version=KEYED_PRG_VERSION,
    )[0]
    original = {
        "latent": torch.zeros_like(reference),
        "attentions": (observed_attention, second_observed_attention),
    }
    aligned = {
        "latent": 0.8 * lf_template + 0.4 * tail_template,
        "attentions": (canonical_attention, second_canonical_attention),
    }
    extraction_count = 0
    stable_selection_count = 0

    import main.methods.detection.image_only as detector_module

    original_select_stable_tokens = detector_module.select_stable_attention_tokens
    original_build_low_frequency_template = (
        detector_module.build_low_frequency_template
    )
    consumed_lf_protocols: list[dict[str, object]] = []

    def build_low_frequency_template_with_trace(
        reference_latent: object,
        consumed_key: str,
        consumed_model: str,
        low_frequency_config: LowFrequencyCarrierConfig,
        **kwargs: object,
    ) -> object:
        """记录 raw 与 aligned 两次盲检实际消费的 LF 协议."""

        consumed_lf_protocols.append(low_frequency_config.to_record())
        return original_build_low_frequency_template(
            reference_latent,
            consumed_key,
            consumed_model,
            low_frequency_config,
            **kwargs,
        )

    monkeypatch.setattr(
        detector_module,
        "build_low_frequency_template",
        build_low_frequency_template_with_trace,
    )

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
        relations = sample["attentions"]
        assert isinstance(relations, tuple) and len(relations) == 2
        relation, second_relation = relations
        assert isinstance(relation, QKAttentionRelation)
        assert isinstance(second_relation, QKAttentionRelation)
        return (
            (layer_name, relation, tuple(range(token_count))),
            (
                second_layer_name,
                second_relation,
                tuple(range(token_count)),
            ),
        )

    result = detect_image_only_watermark(
        image=original,
        key_material=key_material,
        config=_image_only_detection_config(
            model_id=model_id,
            content_threshold=0.2,
            geometry_score_threshold=0.5,
            attention_anchor_count=12,
            attention_residual_threshold=0.20,
            attention_minimum_inlier_ratio=0.50,
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
    assert consumed_lf_protocols == [
        _FORMAL_LOW_FREQUENCY_PROTOCOL,
        _FORMAL_LOW_FREQUENCY_PROTOCOL,
    ]
    assert result.lf_carrier_protocol_digest == (
        _FORMAL_LOW_FREQUENCY_CONFIG.protocol_digest
    )
    assert result.raw_attention_geometry_score is not None
    assert result.attention_geometry_score is not None
    assert result.attention_geometry_score > 0.65
    assert result.attention_sync_score is not None and result.attention_sync_score > 0.65
    assert result.metadata["attention_relation_direct_qk_source_ready"] is True
    assert result.geometry_reliable is True
    assert result.rescue_applied is True
    assert result.metadata["stable_pair_weight_identity_ready"] is True
    assert len(result.metadata["stable_pair_weight_identity_digest"]) == 64
    assert result.metadata["detection_qk_atomic_content_ready"] is True
    assert tuple(
        record["qk_evaluation_role"]
        for record in result.metadata["detection_qk_atomic_content_records"]
    ) == ("raw_detection_image", "aligned_detection_image")
    assert len(result.metadata["detection_qk_atomic_content_digest"]) == 64
    assert result.alignment is not None
    assert result.alignment.attention_relation_component_weights == (
        1.0 / 3.0,
        0.0,
        1.0 / 3.0,
        1.0 / 3.0,
    )
    record = result.to_record()
    assert record["metadata"]["attention_alignment_gate"] == (
        _FORMAL_ATTENTION_ALIGNMENT_GATE
    )
    assert record["alignment"] is not None
    assert {
        field_name: record["alignment"][field_name]
        for field_name in _FORMAL_ATTENTION_ALIGNMENT_GATE
    } == _FORMAL_ATTENTION_ALIGNMENT_GATE
    assert record["alignment"]["metadata"][
        "attention_alignment_gate"
    ] == _FORMAL_ATTENTION_ALIGNMENT_GATE


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
    noise_evidence_records = getattr(
        extractor,
        "public_detection_noise_evidence_records",
    )
    assert len(noise_evidence_records) == 1
    assert noise_evidence_records[0]["tensor_content_digest_version"] == (
        TENSOR_CONTENT_DIGEST_VERSION
    )
    assert len(
        noise_evidence_records[0][
            "public_detection_noise_content_sha256"
        ]
    ) == 64
    relation_identity = build_attention_relation_graph_identity(
        records,
        "test_detection_key",
        prg_version=KEYED_PRG_VERSION,
    )
    detection_record = bind_formal_detection_record(
        {
            "content_score": 0.1,
            "aligned_content_score": None,
            "attention_geometry_score": 0.1,
            "registration_confidence": 0.1,
            "attention_sync_score": 0.1,
            "alignment": None,
        }
    )
    detection_record["metadata"][
        "detection_qk_atomic_content_records"
    ] = [
        {
            "qk_evaluation_role": "raw_detection_image",
            "qk_atomic_content_records": list(
                relation_identity.qk_atomic_content_records
            ),
            "qk_atomic_content_digest": (
                relation_identity.qk_atomic_content_digest
            ),
            "qk_atomic_content_ready": (
                relation_identity.qk_atomic_content_ready
            ),
        }
    ]
    _bind_public_detection_noise_qk_evidence(
        detection_record,
        extractor,
        0,
    )
    assert detection_record["metadata"][
        "public_detection_noise_evidence_ready"
    ] is True
    assert detection_record["metadata"][
        "detection_qk_atomic_content_records"
    ][0]["public_detection_noise_content_sha256"] == (
        detection_record["public_detection_noise_content_sha256"]
    )
    validate_image_only_detection_digest_record(detection_record)
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
    """运行时不得通过别名字段改写冻结的相邻 scheduler 时刻."""

    base = SemanticWatermarkRuntimeConfig()
    with pytest.raises(ValueError, match="唯一正式配置"):
        replace(base, injection_step_indices=(base.inference_steps - 1,))
    with pytest.raises(ValueError, match="唯一正式配置"):
        replace(base, injection_step_indices=(0,))


@pytest.mark.quick
def test_image_only_detector_interface_and_positive_content_path() -> None:
    """正式检测接口不得接收生成轨迹, 且能从图像编码 latent 完成内容主判。"""

    parameters = set(inspect.signature(detect_image_only_watermark).parameters)
    assert "generation_latent_trace" not in parameters
    assert "source_latent" not in parameters
    assert "prompt" not in parameters

    reference = torch.zeros(1, 2, 8, 8)
    lf_template = build_low_frequency_template(
        reference,
        "blind_key",
        "model",
        _FORMAL_LOW_FREQUENCY_CONFIG,
        prg_version=KEYED_PRG_VERSION,
    )
    tail_template = build_tail_robust_template(
        reference,
        "blind_key",
        "model",
        0.20,
        prg_version=KEYED_PRG_VERSION,
    )[0]
    encoded = 0.8 * lf_template + 0.4 * tail_template
    result = detect_image_only_watermark(
        image=encoded,
        key_material="blind_key",
        config=_image_only_detection_config(
            model_id="model",
            content_threshold=0.20,
            geometry_score_threshold=0.0,
            attention_anchor_count=12,
            attention_residual_threshold=0.20,
            attention_minimum_inlier_ratio=0.50,
        ),
        image_latent_encoder=lambda image: image,
    )

    assert result.positive_by_content is True
    assert result.evidence_positive is True
    assert result.content_failure_reason == "content_positive"
    assert result.rescue_applied is False
    assert result.metadata["blind_image_detector"] is True
    assert result.metadata["generation_latent_trace_required"] is False
    assert result.metadata["attention_alignment_gate"] == (
        _FORMAL_ATTENTION_ALIGNMENT_GATE
    )
    assert result.lf_carrier_protocol_digest == (
        _FORMAL_LOW_FREQUENCY_CONFIG.protocol_digest
    )
    assert result.content.lf_weight == _FORMAL_LF_WEIGHT
    assert result.content.tail_robust_weight == _FORMAL_TAIL_ROBUST_WEIGHT


@pytest.mark.quick
@pytest.mark.parametrize("disabled_branch", ("lf_content", "tail_robust"))
def test_disabled_detector_carrier_is_not_built_or_scored(
    monkeypatch: pytest.MonkeyPatch,
    disabled_branch: str,
) -> None:
    """内容分支消融必须跳过模板构造, 并将对应检测原子精确清空."""

    import main.methods.detection.image_only as detector_module

    def fail_builder(*_args: object, **_kwargs: object) -> object:
        """已禁用模板构造器一旦执行就使测试失败."""

        pytest.fail("已禁用内容分支仍构造检测模板")

    if disabled_branch == "lf_content":
        monkeypatch.setattr(
            detector_module,
            "build_low_frequency_template",
            fail_builder,
        )
        lf_weight, tail_weight = 0.0, 1.0
    else:
        monkeypatch.setattr(
            detector_module,
            "build_tail_robust_template",
            fail_builder,
        )
        lf_weight, tail_weight = 1.0, 0.0

    result = detect_image_only_watermark(
        image=torch.zeros(1, 2, 8, 8),
        key_material="disabled_detector_carrier_key",
        config=_image_only_detection_config(
            lf_weight=lf_weight,
            tail_robust_weight=tail_weight,
        ),
        image_latent_encoder=lambda value: value,
    )
    record = result.to_record()

    if disabled_branch == "lf_content":
        assert result.content.lf_score == 0.0
        assert record["lf_template_content_sha256"] == ""
    else:
        assert result.content.tail_robust_score == 0.0
        assert record["tail_template_content_sha256"] == ""
        assert record["tail_template_shape"] == []
        assert record["tail_template_element_count"] == 0
        assert record["tail_selected_element_count"] == 0
    validate_image_only_detection_digest_record(record)


@pytest.mark.quick
def test_alignment_ablation_keeps_raw_qk_and_skips_affine_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关闭图像 alignment 时应保留原图 Q/K 盲检, 但不得恢复仿射变换."""

    import main.methods.detection.image_only as detector_module

    def fail_alignment(*_args: object, **_kwargs: object) -> object:
        """仿射恢复一旦被调用就使 alignment 消融测试失败."""

        pytest.fail("image_aligner=None 时仍执行仿射恢复")

    monkeypatch.setattr(
        detector_module,
        "recover_attention_affine_alignment",
        fail_alignment,
    )
    layer_names = FROZEN_SD35_ATTENTION_MODULE_NAMES
    relations = tuple(
        _direct_qk_relation_from_logits(
            torch.randn(
                1,
                16,
                16,
                generator=torch.Generator().manual_seed(260718 + index),
            ),
            layer_name,
        )
        for index, layer_name in enumerate(layer_names)
    )
    extraction_count = 0

    def extract(_image: object) -> tuple[tuple[str, object, tuple[int, ...]], ...]:
        """返回一次具有冻结层身份的原图 Q/K 关系."""

        nonlocal extraction_count
        extraction_count += 1
        return tuple(
            (layer_name, relation, tuple(range(16)))
            for layer_name, relation in zip(layer_names, relations)
        )

    result = detect_image_only_watermark(
        image=torch.zeros(1, 2, 8, 8),
        key_material="alignment_ablation_key",
        config=_image_only_detection_config(),
        image_latent_encoder=lambda value: value,
        image_attention_extractor=extract,
        image_aligner=None,
    )
    qk_roles = tuple(
        item["qk_evaluation_role"]
        for item in result.metadata["detection_qk_atomic_content_records"]
    )

    assert extraction_count == 1
    assert result.raw_attention_geometry_score is not None
    assert result.attention_geometry_score is None
    assert result.alignment is None
    assert result.metadata["image_alignment_enabled"] is False
    assert qk_roles == ("raw_detection_image",)
    validate_image_only_detection_digest_record(result.to_record())


@pytest.mark.quick
def test_detector_digest_binds_every_alignment_gate_parameter() -> None:
    """内容主判路径的检测摘要也必须逐字段绑定注意力结构门禁."""

    image = torch.zeros(1, 2, 8, 8)
    baseline_config = _image_only_detection_config(
        model_id="detector_gate_digest_model",
        content_threshold=0.20,
        geometry_score_threshold=0.0,
        attention_anchor_count=12,
        attention_residual_threshold=0.20,
        attention_minimum_inlier_ratio=0.50,
    )
    baseline = detect_image_only_watermark(
        image=image,
        key_material="detector_gate_digest_key",
        config=baseline_config,
        image_latent_encoder=lambda value: value,
    )
    for field_name, value in (
        ("attention_anchor_count", 13),
        ("attention_residual_threshold", 0.21),
        ("attention_minimum_inlier_ratio", 0.51),
    ):
        changed_config = replace(
            baseline_config,
            **{field_name: value},
        )
        changed = detect_image_only_watermark(
            image=image,
            key_material="detector_gate_digest_key",
            config=changed_config,
            image_latent_encoder=lambda candidate: candidate,
        )
        assert changed.detector_digest != baseline.detector_digest
        assert (
            changed.image_only_detector_config_digest
            != baseline.image_only_detector_config_digest
        )


@pytest.mark.quick
def test_detector_rejects_alignment_selected_from_unfrozen_layer() -> None:
    """可重算的 alignment 仍必须来自冻结 SD3.5 层集合."""

    metadata, alignment = _formal_detection_alignment_identity(
        registration_geometry_reliable=True,
    )
    record = bind_formal_detection_record(
        {
            **_formal_content_carrier_identity_fields(),
            "content_score": 0.10,
            "aligned_content_score": 0.20,
            "attention_geometry_score": 0.30,
            "registration_confidence": 0.40,
            "attention_sync_score": 0.50,
            "metadata": metadata,
            "alignment": alignment,
        }
    )
    record["alignment"]["layer_name"] = "transformer_blocks.1.attn"
    record["alignment"]["alignment_digest"] = build_stable_digest(
        recompute_attention_alignment_digest_payload(record["alignment"])
    )
    record["detector_digest"] = build_stable_digest(
        recompute_image_only_detection_digest_payload(record)
    )

    with pytest.raises(ValueError, match="所选层不属于冻结"):
        validate_image_only_detection_digest_record(record)


@pytest.mark.quick
def test_identity_alignment_cannot_propagate_into_detector_rescue() -> None:
    """identity 配准即使内容 margin 位于救回窗口也不得开放 rescue。"""

    metadata, alignment = _formal_detection_alignment_identity(
        registration_geometry_reliable=False,
    )
    metadata.update(
        {
            "content_threshold": 0.20,
            "geometry_score_threshold": 0.50,
            "registration_confidence_threshold": 0.50,
            "attention_sync_score_threshold": 0.50,
            "rescue_margin_low": -0.05,
        }
    )
    record = bind_formal_detection_record(
        {
            **_formal_content_carrier_identity_fields(),
            "content_score": 0.18,
            "aligned_content_score": 0.25,
            "attention_geometry_score": 0.90,
            "registration_confidence": 0.90,
            "attention_sync_score": 0.90,
            "metadata": metadata,
            "alignment": alignment,
        }
    )

    assert record["raw_content_margin"] == pytest.approx(-0.02)
    assert record["alignment"]["affine_transform"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    assert record["alignment"]["registration_objective_score"] == pytest.approx(
        record["alignment"]["identity_registration_objective_score"]
    )
    assert record["alignment"]["registration_objective_margin"] == 0.0
    assert record["alignment"]["geometry_reliable"] is False
    assert record["geometry_reliable"] is False
    assert record["rescue_eligible"] is False
    assert record["rescue_applied"] is False
    assert record["evidence_positive"] is False
    validate_image_only_detection_digest_record(record)

    forged = deepcopy(record)
    forged["alignment"]["geometry_reliable"] = True
    forged["alignment"]["registration_geometry_reliable"] = True
    forged["alignment"]["alignment_digest"] = build_stable_digest(
        recompute_attention_alignment_digest_payload(forged["alignment"])
    )
    forged.update(
        {
            "geometry_reliable": True,
            "content_failure_reason": "geometry_suspected",
            "rescue_eligible": True,
            "rescue_applied": True,
            "evidence_positive": True,
        }
    )
    forged["detector_digest"] = build_stable_digest(
        recompute_image_only_detection_digest_payload(forged)
    )
    with pytest.raises(ValueError, match="注册可靠性与核心门禁不一致"):
        validate_image_only_detection_digest_record(forged)


@pytest.mark.quick
def test_detector_digest_separates_zero_score_template_content_collisions() -> None:
    """相同零分数摘要不得掩盖不同 shape 对应的 LF 模板身份."""

    config = _image_only_detection_config(
        model_id="detector_zero_score_collision_model"
    )
    first = detect_image_only_watermark(
        image=torch.zeros(1, 1, 8, 8),
        key_material="detector_zero_score_collision_key",
        config=config,
        image_latent_encoder=lambda value: value,
    )
    second = detect_image_only_watermark(
        image=torch.zeros(1, 2, 8, 8),
        key_material="detector_zero_score_collision_key",
        config=config,
        image_latent_encoder=lambda value: value,
    )

    assert first.content.content_score == second.content.content_score == 0.0
    assert first.content.score_digest == second.content.score_digest
    assert first.lf_template_content_sha256 != second.lf_template_content_sha256
    assert first.detector_digest != second.detector_digest


@pytest.mark.quick
def test_complete_evidence_calibration_includes_geometry_rescue() -> None:
    """阈值搜索必须直接约束加入同阈值 rescue 后的完整误报率。"""

    calibration_records = []
    for index in range(33):
        metadata, alignment = _formal_detection_alignment_identity(
            registration_geometry_reliable=index % 2 == 0,
        )
        calibration_records.append(
            bind_formal_detection_record({
                **_formal_content_carrier_identity_fields(),
                "content_score": index / 100.0,
                "aligned_content_score": (index + 5) / 100.0,
                "geometry_reliable": index % 2 == 0,
                "attention_geometry_score": 0.5 + index / 1000.0,
                "registration_confidence": 0.6 + index / 1000.0,
                "attention_sync_score": 0.7 + index / 1000.0,
                "metadata": metadata,
                "alignment": alignment,
            })
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
def test_frozen_rescue_rejects_unbound_stable_pair_identity() -> None:
    """冻结 rescue 不得绕过检测器的稳定 pair 权重身份门禁."""

    decision = _decision(
        {
            "content_score": 0.10,
            "aligned_content_score": 0.20,
            "attention_geometry_score": 0.90,
            "registration_confidence": 0.90,
            "attention_sync_score": 0.90,
            "alignment": {
                "registration_geometry_reliable": True,
                "geometry_reliable": True,
            },
            "metadata": {
                "stable_pair_weight_identity_ready": False,
            },
        },
        threshold=0.15,
        rescue_margin_low=-0.10,
        geometry_score_threshold=0.50,
        registration_confidence_threshold=0.50,
        attention_sync_score_threshold=0.50,
    )

    assert decision == (False, False, False, "low_confidence")


@pytest.mark.quick
def test_frozen_evidence_protocol_rejects_alignment_gate_drift() -> None:
    """校准和应用环节都不得接受非预注册注意力结构门禁."""

    metadata, alignment = _formal_detection_alignment_identity(
        registration_geometry_reliable=True,
    )
    baseline_record = bind_formal_detection_record({
        **_formal_content_carrier_identity_fields(),
        "content_score": 0.1,
        "aligned_content_score": 0.2,
        "attention_geometry_score": 0.3,
        "registration_confidence": 0.4,
        "attention_sync_score": 0.5,
        "metadata": metadata,
        "alignment": alignment,
    })
    protocol = calibrate_complete_evidence_protocol(
        (baseline_record,),
        target_fpr=0.1,
        rescue_margin_low=-0.05,
    )
    changed_record = deepcopy(baseline_record)
    changed_value = 0.21
    changed_record["metadata"][
        "attention_residual_threshold"
    ] = changed_value
    changed_record["metadata"]["attention_alignment_gate"][
        "attention_residual_threshold"
    ] = changed_value
    changed_record["alignment"][
        "attention_residual_threshold"
    ] = changed_value
    changed_record["alignment"]["metadata"]["attention_alignment_gate"][
        "attention_residual_threshold"
    ] = changed_value

    with pytest.raises(ValueError, match="预注册注意力结构门禁"):
        calibrate_complete_evidence_protocol(
            (changed_record,),
            target_fpr=0.1,
            rescue_margin_low=-0.05,
        )
    with pytest.raises(ValueError, match="预注册注意力结构门禁"):
        apply_frozen_evidence_protocol((changed_record,), protocol)


@pytest.mark.quick
def test_frozen_evidence_protocol_rejects_low_frequency_protocol_drift() -> None:
    """校准和应用均不得接受 LF 协议摘要或权重分叉."""

    metadata, alignment = _formal_detection_alignment_identity(
        registration_geometry_reliable=True,
    )
    baseline_record = bind_formal_detection_record({
        **_formal_content_carrier_identity_fields(),
        "content_score": 0.1,
        "aligned_content_score": 0.2,
        "attention_geometry_score": 0.3,
        "registration_confidence": 0.4,
        "attention_sync_score": 0.5,
        "metadata": metadata,
        "alignment": alignment,
    })
    protocol = calibrate_complete_evidence_protocol(
        (baseline_record,),
        target_fpr=0.1,
        rescue_margin_low=-0.05,
    )
    for mutation in ("digest", "weight"):
        changed_record = deepcopy(baseline_record)
        if mutation == "digest":
            changed_record["lf_carrier_protocol_digest"] = "0" * 64
        else:
            changed_record["lf_weight"] = 0.69
            changed_record["tail_robust_weight"] = 0.31
        changed_record["detector_digest"] = build_stable_digest(
            recompute_image_only_detection_digest_payload(changed_record)
        )

        with pytest.raises(ValueError):
            calibrate_complete_evidence_protocol(
                (changed_record,),
                target_fpr=0.1,
                rescue_margin_low=-0.05,
            )
        with pytest.raises(ValueError):
            apply_frozen_evidence_protocol((changed_record,), protocol)


@pytest.mark.quick
def test_threshold_digest_binds_every_alignment_gate_parameter() -> None:
    """冻结阈值摘要必须逐字段绑定结构门禁而非只绑定分数阈值."""

    metadata, alignment = _formal_detection_alignment_identity(
        registration_geometry_reliable=True,
    )
    protocol = calibrate_complete_evidence_protocol(
        (
            bind_formal_detection_record({
                **_formal_content_carrier_identity_fields(),
                "content_score": 0.1,
                "aligned_content_score": 0.2,
                "attention_geometry_score": 0.3,
                "registration_confidence": 0.4,
                "attention_sync_score": 0.5,
                "metadata": metadata,
                "alignment": alignment,
            }),
        ),
        target_fpr=0.1,
        rescue_margin_low=-0.05,
    )
    digest_payload = {
        field_name: value
        for field_name, value in protocol.to_dict().items()
        if field_name not in {
            "calibration_false_positive_rate",
            "threshold_digest",
        }
    }
    digest_payload["decision_scope"] = (
        "content_or_same_threshold_aligned_content_rescue"
    )
    assert build_stable_digest(digest_payload) == protocol.threshold_digest
    for field_name, value in (
        ("attention_anchor_count", 13),
        ("attention_residual_threshold", 0.21),
        ("attention_minimum_inlier_ratio", 0.51),
        ("lf_carrier_protocol_digest", "0" * 64),
        ("lf_weight", 0.69),
        ("tail_robust_weight", 0.31),
        ("tail_fraction", 1.0),
        ("tail_carrier_protocol_digest", "0" * 64),
    ):
        changed_payload = dict(digest_payload)
        changed_payload[field_name] = value
        assert build_stable_digest(changed_payload) != protocol.threshold_digest


@pytest.mark.quick
def test_frozen_protocol_application_rejects_threshold_digest_drift() -> None:
    """协议应用时必须先从全部冻结字段独立重建阈值摘要."""

    metadata, alignment = _formal_detection_alignment_identity(
        registration_geometry_reliable=True,
    )
    record = bind_formal_detection_record(
        {
            **_formal_content_carrier_identity_fields(),
            "content_score": 0.1,
            "aligned_content_score": 0.2,
            "attention_geometry_score": 0.3,
            "registration_confidence": 0.4,
            "attention_sync_score": 0.5,
            "metadata": metadata,
            "alignment": alignment,
        }
    )
    protocol = calibrate_complete_evidence_protocol(
        (record,),
        target_fpr=0.1,
        rescue_margin_low=-0.05,
    )
    for drifted in (
        replace(protocol, content_threshold=999.0),
        replace(protocol, threshold_digest="f" * 64),
        replace(protocol, calibration_false_positive_rate=0.999),
    ):
        with pytest.raises(
            ValueError,
            match="阈值摘要不能由正文重建|假阳性率与计数不一致",
        ):
            apply_frozen_evidence_protocol((record,), drifted)


@pytest.mark.quick
def test_geometry_protocol_cannot_close_with_missing_calibration_scores() -> None:
    """任一几何数值门禁缺失时不得把完整 rescue 协议标记为已校准。"""

    records = []
    for index in range(33):
        metadata, alignment = _formal_detection_alignment_identity(
            registration_geometry_reliable=True,
        )
        records.append(
            bind_formal_detection_record({
                **_formal_content_carrier_identity_fields(),
                "content_score": index / 100.0,
                "aligned_content_score": (index + 1) / 100.0,
                "attention_geometry_score": 0.1,
                "registration_confidence": 0.2,
                "attention_sync_score": None if index == 0 else 0.3,
                "metadata": metadata,
                "alignment": alignment,
            })
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

    metadata, alignment = _formal_detection_alignment_identity(
        registration_geometry_reliable=True,
        geometry_reliable=True,
    )
    metadata["rescue_margin_low"] = -0.2
    record = bind_formal_detection_record({
        **_formal_content_carrier_identity_fields(),
        "content_score": 0.4,
        "aligned_content_score": 0.6,
        "attention_geometry_score": 0.1,
        "registration_confidence": 0.8,
        "attention_sync_score": 0.8,
        "geometry_reliable": False,
        "metadata": metadata,
        "alignment": alignment,
        "content_failure_reason": "content_positive",
    })
    protocol = FrozenEvidenceProtocol(
        content_threshold=0.5,
        rescue_margin_low=-0.2,
        geometry_score_threshold=0.0,
        registration_confidence_threshold=0.0,
        attention_sync_score_threshold=0.0,
        attention_anchor_count=ATTENTION_ALIGNMENT_ANCHOR_COUNT,
        attention_residual_threshold=ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD,
        attention_minimum_inlier_ratio=(
            ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO
        ),
        lf_carrier_protocol_digest=(
            _FORMAL_LOW_FREQUENCY_CONFIG.protocol_digest
        ),
        lf_weight=_FORMAL_LF_WEIGHT,
        tail_robust_weight=_FORMAL_TAIL_ROBUST_WEIGHT,
        tail_fraction=_FORMAL_TAIL_FRACTION,
        tail_carrier_protocol_digest=_FORMAL_TAIL_CARRIER_PROTOCOL[
            "tail_carrier_protocol_digest"
        ],
        image_only_detector_config_digest=record[
            "image_only_detector_config_digest"
        ],
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
    protocol = replace(
        protocol,
        threshold_digest=build_stable_digest(
            frozen_evidence_protocol_digest_payload(protocol)
        ),
    )
    resolved = apply_frozen_evidence_protocol((record,), protocol)[0]

    assert resolved["formal_content_failure_reason"] == "geometry_suspected"
    assert resolved["formal_positive_by_content"] is False
    assert resolved["formal_rescue_applied"] is True
    assert resolved["formal_evidence_positive"] is True


@pytest.mark.quick
def test_completed_runtime_cache_rejects_missing_scientific_content_binding(
    tmp_path: Path,
) -> None:
    """Colab 续跑必须拒绝缺少总科学内容绑定的旧运行结果。"""

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

    assert (
        load_completed_semantic_watermark_runtime_result(
            config,
            root=tmp_path,
        )
        is None
    )


@pytest.mark.quick
def test_closed_archive_recovery_without_directories_is_empty(
    tmp_path: Path,
) -> None:
    """未配置外部归档目录时恢复路径必须保持无操作."""

    recovered = scientific_workflow._recover_closed_archives(
        root_path=tmp_path,
        paper_run_name="probe_paper",
        target_fpr=0.1,
        randomization_repeat_id="seed_00_key_00",
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
    write_formal_method_config(tmp_path)
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

    write_formal_method_config(tmp_path)
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
                randomization_repeat_id="seed_00_key_00",
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
    write_formal_method_config(tmp_path)
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
    write_formal_method_config(tmp_path)
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
                "formal_fid_kid_component_ready": True,
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
    write_formal_method_config(tmp_path)
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
        json.dumps({"formal_fid_kid_component_ready": True}),
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
