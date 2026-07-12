"""验证完整特征 Jacobian 与累计成图保持门禁。"""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.runtime.diffusion.semantic_features import (
    JOINT_FEATURE_WIDTH,
    SEMANTIC_FEATURE_SCHEMA,
    SEMANTIC_FEATURE_WIDTH,
    VISUAL_FEATURE_SCHEMA,
    VISUAL_FEATURE_WIDTH,
    DifferentiableSemanticFeatureRuntime,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    _carrier_only_counterfactual_artifact_binding_ready,
    _carrier_only_counterfactual_identity,
    _combined_update_preservation_record,
    _final_image_attention_attribution_gate_ready,
    _final_image_attention_observability_record,
    _final_image_preservation_record,
    _tensor_content_sha256,
    _three_way_final_image_preservation_records,
)
from experiments.runners.image_only_dataset_runtime import (
    _final_image_attention_observability_ready,
)
from experiments.runtime.repository_environment import file_digest
from main.methods.geometry.differentiable_attention import (
    QKAttentionRelation,
    keyed_relation_signs,
)
from main.methods.subspace import build_exact_jacobian_linearization


def _counterfactual_update_records(
    config: SemanticWatermarkRuntimeConfig,
    *,
    role: str,
    attention_enabled: bool,
) -> list[dict[str, object]]:
    """构造满足 carrier-only 原子协议的轻量更新记录。"""

    branches = ["lf_content", "tail_robust"]
    if attention_enabled:
        branches.append("attention_geometry")
    records = []
    for record_index, step_index in enumerate(config.injection_step_indices):
        before_sha = "1" * 64 if record_index == 0 else "2" * 64
        attention_record = (
            {
                "attention_score_before": 0.1,
                "attention_content_base_score": 0.1,
                "attention_score_after": 0.2,
                "attention_final_combined_score": 0.2,
                "attention_score_gain": 0.1,
                "attention_applied_update_strength": 0.01,
                "attention_backtracking_step_count": 0,
                "attention_update_digest": "3" * 64,
                "stable_token_indices": [0, 1, 2, 3],
                "stable_token_selection_digest": "4" * 64,
                "stable_pair_weight_identity_digest": "5" * 64,
                "stable_pair_weight_realization_digest": "6" * 64,
                "attention_relation_component_names": ["direct_qk_fixture"],
                "attention_relation_source": "direct_qk_fixture",
                "attention_relation_direct_qk_source_ready": True,
                "attention_relation_probability_scope": "fixture_probability",
                "attention_relation_component_identity_digest": "7" * 64,
                "attention_relation_keyed_projection_digest": "8" * 64,
                "attention_relation_qk_operator_metadata_records": [
                    {"operator_identity": "fixture"}
                ],
                "attention_relation_qk_operator_metadata_digest": "b" * 64,
                "attention_relation_qk_operator_metadata_ready": True,
            }
            if attention_enabled
            else {
                "attention_score_before": None,
                "attention_content_base_score": None,
                "attention_score_after": None,
                "attention_final_combined_score": None,
                "attention_score_gain": None,
                "attention_applied_update_strength": None,
                "attention_backtracking_step_count": None,
                "attention_update_digest": "",
                "stable_token_indices": [],
                "stable_token_selection_digest": "",
                "stable_pair_weight_identity_digest": "",
                "stable_pair_weight_realization_digest": "",
                "attention_relation_component_names": [],
                "attention_relation_source": "",
                "attention_relation_direct_qk_source_ready": False,
                "attention_relation_probability_scope": "",
                "attention_relation_component_identity_digest": "",
                "attention_relation_keyed_projection_digest": "",
                "attention_relation_qk_operator_metadata_records": [],
                "attention_relation_qk_operator_metadata_digest": "",
                "attention_relation_qk_operator_metadata_ready": False,
            }
        )
        records.append(
            {
                "step_index": step_index,
                "scheduler_step_timestep": float(100 - step_index),
                "post_step_schedule_index": step_index + 1,
                "timestep": float(99 - step_index),
                "latent_content_sha256_before": before_sha,
                "latent_content_sha256_after": "9" * 64,
                "combined_update_content_sha256": "a" * 64,
                "active_carrier_branches": list(branches),
                "null_space_records": {
                    branch_name: {"branch_name": branch_name}
                    for branch_name in branches
                },
                "metadata": {
                    "injection_execution_role": role,
                    "attention_geometry_enabled": attention_enabled,
                    "attention_source": (
                        "real_qk_projection"
                        if attention_enabled
                        else "disabled_attention_geometry"
                    ),
                },
                **attention_record,
            }
        )
    return records


@pytest.mark.quick
def test_formal_jacobian_vector_keeps_every_clip_and_visual_coordinate() -> None:
    """正式 Jacobian 输入必须直接连接512维 CLIP 与204维视觉特征。"""

    semantic = torch.arange(SEMANTIC_FEATURE_WIDTH, dtype=torch.float32).reshape(1, -1)
    visual = torch.arange(VISUAL_FEATURE_WIDTH, dtype=torch.float32)
    runtime = SimpleNamespace(joint_features=lambda _latent: (semantic, visual))

    vector = DifferentiableSemanticFeatureRuntime.full_joint_feature_vector(
        runtime,
        torch.zeros(1, 1, 1, 1),
    )

    assert vector.shape == (JOINT_FEATURE_WIDTH,)
    assert torch.equal(vector[:SEMANTIC_FEATURE_WIDTH], semantic.reshape(-1))
    assert torch.equal(vector[SEMANTIC_FEATURE_WIDTH:], visual)


@pytest.mark.quick
def test_full_feature_schema_declares_no_compression() -> None:
    """完整特征 schema 必须冻结宽度并显式声明未压缩。"""

    runtime = object.__new__(DifferentiableSemanticFeatureRuntime)
    object.__setattr__(
        runtime,
        "vision_model",
        SimpleNamespace(config=SimpleNamespace(projection_dim=SEMANTIC_FEATURE_WIDTH)),
    )

    record = runtime.feature_schema_record()

    assert record == {
        "semantic_feature_schema": SEMANTIC_FEATURE_SCHEMA,
        "semantic_feature_width": SEMANTIC_FEATURE_WIDTH,
        "visual_feature_schema": VISUAL_FEATURE_SCHEMA,
        "visual_feature_width": VISUAL_FEATURE_WIDTH,
        "joint_feature_width": JOINT_FEATURE_WIDTH,
        "feature_compression_applied": False,
    }


@pytest.mark.quick
def test_full_visual_vector_preserves_spatial_and_gradient_information() -> None:
    """完整视觉向量必须保留通道统计、梯度与8x8空间池化。"""

    smooth = torch.full((1, 3, 16, 16), 0.5)
    checker = smooth.clone()
    checker[:, :, ::2, ::2] = 1.0
    checker[:, :, 1::2, 1::2] = 0.0
    spatial = smooth.clone()
    spatial[:, :, :, :8] = 0.25
    spatial[:, :, :, 8:] = 0.75

    smooth_features = DifferentiableSemanticFeatureRuntime._visual_features_from_image(
        smooth
    )
    checker_features = DifferentiableSemanticFeatureRuntime._visual_features_from_image(
        checker
    )
    spatial_features = DifferentiableSemanticFeatureRuntime._visual_features_from_image(
        spatial
    )

    assert smooth_features.shape == (VISUAL_FEATURE_WIDTH,)
    assert checker_features.shape == (VISUAL_FEATURE_WIDTH,)
    assert checker_features[3:6].mean() > smooth_features[3:6].mean()
    assert checker_features[6:9].mean() > smooth_features[6:9].mean()
    assert checker_features[9:12].mean() > smooth_features[9:12].mean()
    assert not torch.equal(
        spatial_features[12:],
        smooth_features[12:],
    )


@pytest.mark.quick
def test_complete_feature_vector_supports_exact_jvp_and_vjp() -> None:
    """716维完整输出必须同时支持精确 JVP 与 VJP。"""

    latent = torch.linspace(-1.0, 1.0, 16)
    projection = torch.arange(
        JOINT_FEATURE_WIDTH * latent.numel(),
        dtype=torch.float32,
    ).reshape(JOINT_FEATURE_WIDTH, latent.numel())
    projection = projection.remainder(17.0) / 17.0

    def full_features(values: torch.Tensor) -> torch.Tensor:
        return projection @ values

    linearization = build_exact_jacobian_linearization(full_features, latent)
    tangent = linearization.apply(torch.ones_like(latent))
    cotangent = linearization.transpose_apply(torch.ones(JOINT_FEATURE_WIDTH))

    assert linearization.output_width == JOINT_FEATURE_WIDTH
    assert tangent.shape == (JOINT_FEATURE_WIDTH,)
    assert cotangent.shape == latent.shape
    assert torch.allclose(tangent, projection @ torch.ones_like(latent))
    assert torch.allclose(cotangent, projection.transpose(0, 1) @ torch.ones(JOINT_FEATURE_WIDTH))


@pytest.mark.quick
def test_actual_combined_latent_uses_full_feature_preservation_gate() -> None:
    """有限更新门禁必须检查完整特征, 而不只信局部 Jacobian 残差。"""

    class _FeatureRuntime:
        """提供可精确控制的完整语义与视觉特征。"""

        @staticmethod
        def joint_features(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            values = latent.reshape(-1).float()
            semantic = torch.nn.functional.normalize(values[:2], dim=0)
            visual = torch.stack((values[0], values[1], values.square().mean()))
            return semantic, visual

    config = SemanticWatermarkRuntimeConfig(
        minimum_semantic_preservation_cosine=0.99,
        maximum_visual_feature_relative_drift=0.05,
    )
    latent = torch.tensor([1.0, 0.0, 0.5])
    accepted = _combined_update_preservation_record(
        _FeatureRuntime(),
        latent,
        torch.tensor([1.0, 0.001, 0.5]),
        config,
    )
    rejected = _combined_update_preservation_record(
        _FeatureRuntime(),
        latent,
        torch.tensor([0.0, 1.0, 0.5]),
        config,
    )

    assert accepted["semantic_preservation_gate_ready"] is True
    assert rejected["semantic_preservation_gate_ready"] is False


@pytest.mark.quick
def test_tensor_content_sha256_binds_dtype_shape_and_raw_bytes() -> None:
    """Tensor 身份必须覆盖 dtype、shape 和全部连续原始字节。"""

    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    noncontiguous = values.transpose(0, 1)

    digest = _tensor_content_sha256(values)

    assert len(digest) == 64
    assert digest != _tensor_content_sha256(values.to(dtype=torch.float16))
    assert digest != _tensor_content_sha256(values.reshape(4))
    changed = values.clone()
    changed[0, 0] = 1.0001
    assert digest != _tensor_content_sha256(changed)
    assert _tensor_content_sha256(noncontiguous) == _tensor_content_sha256(
        noncontiguous.contiguous()
    )


@pytest.mark.quick
def test_final_image_gate_checks_cumulative_clean_to_watermarked_drift() -> None:
    """最终门禁必须直接比较 clean 与 watermarked 成图的累计变化。"""

    class _ImageProcessor:
        """模拟正式预处理器返回 [-1, 1] tensor。"""

        @staticmethod
        def preprocess(image: torch.Tensor) -> torch.Tensor:
            return image * 2.0 - 1.0

    class _ImageFeatureRuntime:
        """从最终图像 tensor 生成可控制的语义与视觉特征。"""

        @staticmethod
        def joint_image_features(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            flat = image.reshape(-1).float()
            semantic = torch.nn.functional.normalize(flat[:3], dim=0)
            return semantic, flat

    pipeline = SimpleNamespace(
        _execution_device="cpu",
        image_processor=_ImageProcessor(),
    )
    config = SemanticWatermarkRuntimeConfig(
        minimum_semantic_preservation_cosine=0.99,
        maximum_visual_feature_relative_drift=0.05,
    )
    clean = torch.tensor([[[[1.0, 0.5], [0.25, 0.75]]]])
    close = clean + 0.001
    changed = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])

    accepted = _final_image_preservation_record(
        pipeline,
        _ImageFeatureRuntime(),
        clean,
        close,
        config,
    )
    rejected = _final_image_preservation_record(
        pipeline,
        _ImageFeatureRuntime(),
        clean,
        changed,
        config,
    )

    assert accepted["final_image_preservation_gate_ready"] is True
    assert rejected["final_image_preservation_gate_ready"] is False

    counterfactual = {
        "carrier_only_counterfactual_ready": True,
        "carrier_only_counterfactual_identity_digest": "a" * 64,
    }
    final_accepted, carrier_accepted = (
        _three_way_final_image_preservation_records(
            pipeline,
            _ImageFeatureRuntime(),
            clean,
            close,
            clean + 0.002,
            config,
            counterfactual,
        )
    )
    center = torch.tensor([[[[0.6, 0.5], [0.4, 0.5]]]])
    edge_config = SemanticWatermarkRuntimeConfig(
        minimum_semantic_preservation_cosine=0.99,
        maximum_visual_feature_relative_drift=0.10,
    )
    final_edge, carrier_edge = _three_way_final_image_preservation_records(
        pipeline,
        _ImageFeatureRuntime(),
        center,
        center + 0.04,
        center - 0.04,
        edge_config,
        counterfactual,
    )

    assert final_accepted["final_image_preservation_gate_ready"] is True
    assert carrier_accepted[
        "carrier_only_counterfactual_three_way_preservation_gate_ready"
    ] is True
    assert carrier_accepted["carrier_only_counterfactual_identity_digest"] == (
        "a" * 64
    )
    assert final_edge["final_image_preservation_gate_ready"] is True
    assert carrier_edge[
        "carrier_only_final_image_preservation_gate_ready"
    ] is True
    assert carrier_edge[
        "carrier_only_to_full_final_image_preservation_gate_ready"
    ] is False
    assert carrier_edge[
        "carrier_only_counterfactual_three_way_preservation_gate_ready"
    ] is False
    with pytest.raises(RuntimeError, match="缺少反事实身份"):
        _three_way_final_image_preservation_records(
            pipeline,
            _ImageFeatureRuntime(),
            clean,
            close,
            clean + 0.002,
            config,
            {"carrier_only_counterfactual_ready": True},
        )


@pytest.mark.quick
def test_final_image_attention_gate_uses_reencoded_real_qk_scores() -> None:
    """最终注意力门禁必须同时验证盲选择增益和冻结 pair 权重增益。"""

    token_count = 9
    token_indices = tuple(range(token_count))
    key_material = "final_image_qk_gate_key"
    layer_names = ("final_qk_layer_a", "final_qk_layer_b")

    def records(signature_strength: float):
        """构造可精确控制密钥关系强度的两层真实 Q/K 记录形状。"""

        resolved = []
        for layer_name in layer_names:
            signs = keyed_relation_signs(
                torch.zeros(1, token_count, token_count),
                key_material,
                layer_name,
            )
            logits = (
                signature_strength * signs
                if signature_strength > 0.0
                else torch.zeros(token_count, token_count)
            ).unsqueeze(0)
            attention = QKAttentionRelation(
                centered_logits=logits - logits.mean(dim=-1, keepdim=True),
                probabilities=torch.softmax(logits, dim=-1),
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
                    "source_grid_side": 3,
                    "sampled_token_count": token_count,
                    "sampled_grid_side": 3,
                    "sampled_token_indices": list(token_indices),
                    "centered_logit_aggregation": (
                        "mean_of_per_head_row_centered_sampled_qk_logits"
                    ),
                    "relation_probability_aggregation": (
                        "mean_of_per_head_sampled_image_token_probabilities"
                    ),
                    "mean_probability_is_softmax_of_mean_logits": False,
                },
            )
            resolved.append((layer_name, attention, token_indices))
        return tuple(resolved)

    images = {
        "clean": records(0.0),
        "carrier_only": records(0.0),
        "watermarked": records(2.0),
    }
    counterfactual = {
        "carrier_only_counterfactual_ready": True,
        "carrier_only_counterfactual_changed_fields": [
            "attention_geometry_enabled"
        ],
        "carrier_only_counterfactual_scheduler_identity_ready": True,
        "carrier_only_counterfactual_attention_geometry_enabled": False,
        "carrier_only_counterfactual_identity_digest": "a" * 64,
        "carrier_only_counterfactual_config_digest": "b" * 64,
        "carrier_only_counterfactual_update_records_digest": "c" * 64,
        "carrier_only_counterfactual_scheduler_trace_digest": "d" * 64,
    }
    config = SemanticWatermarkRuntimeConfig(
        key_material=key_material,
        minimum_final_image_attention_score_gain=0.0001,
    )
    accepted = _final_image_attention_observability_record(
        lambda image: images[image],
        "clean",
        "carrier_only",
        "watermarked",
        config,
        carrier_only_counterfactual=counterfactual,
        require_gpu_execution=False,
    )
    rejected = _final_image_attention_observability_record(
        lambda image: images[image],
        "clean",
        "watermarked",
        "carrier_only",
        config,
        carrier_only_counterfactual=counterfactual,
        require_gpu_execution=False,
    )
    drifted_carrier_records = list(records(0.0))
    drifted_layer_name, drifted_relation, drifted_token_indices = (
        drifted_carrier_records[0]
    )
    drifted_carrier_records[0] = (
        drifted_layer_name,
        replace(
            drifted_relation,
            metadata={
                **drifted_relation.metadata,
                "module_class_name": "tests.DriftedDirectQKRelation",
            },
        ),
        drifted_token_indices,
    )
    drifted_images = {
        **images,
        "carrier_only": tuple(drifted_carrier_records),
    }
    with pytest.raises(RuntimeError, match="没有共享直接 Q/K 四分量关系图身份"):
        _final_image_attention_observability_record(
            lambda image: drifted_images[image],
            "clean",
            "carrier_only",
            "watermarked",
            config,
            carrier_only_counterfactual=counterfactual,
            require_gpu_execution=False,
        )

    assert accepted["final_image_attention_observability_gate_ready"] is True
    assert accepted["final_image_attention_blind_attribution_gain"] > 0.0
    assert accepted[
        "final_image_attention_carrier_paired_attribution_gain"
    ] > 0.0
    assert accepted["attention_relation_direct_qk_source_ready"] is True
    assert accepted["attention_relation_qk_operator_metadata_ready"] is True
    assert len(accepted["attention_relation_qk_operator_metadata_records"]) == 2
    assert len(accepted["attention_relation_qk_operator_metadata_digest"]) == 64
    assert isinstance(
        accepted["final_image_attention_carrier_paired_component_gains"][
            "distance_modulated_centered_attention_probability"
        ],
        float,
    )
    assert accepted["final_image_attention_observability_source"] == (
        "image_reencoded_public_noise_real_qk"
    )
    assert accepted["final_image_attention_observability_requires_gpu"] is True
    assert accepted[
        "final_image_attention_observability_gpu_execution_verified"
    ] is False
    assert len(accepted["final_carrier_only_pair_weight_identity_digest"]) == 64
    assert len(accepted["final_image_attention_record_schema_digest"]) == 64
    assert rejected["final_image_attention_observability_gate_ready"] is False
    carrier_image_path = "outputs/runtime/carrier_only_image.png"
    carrier_image_digest = "e" * 64
    accepted["carrier_only_counterfactual_image_path"] = carrier_image_path
    accepted["carrier_only_counterfactual_image_digest"] = carrier_image_digest
    preservation = {
        "carrier_only_final_image_preservation_applicable": True,
        "carrier_only_final_image_preservation_gate_ready": True,
        "carrier_only_final_image_semantic_cosine_similarity": 1.0,
        "carrier_only_final_image_visual_feature_relative_drift": 0.0,
        "carrier_only_counterfactual_identity_digest": "a" * 64,
        "carrier_only_counterfactual_image_path": carrier_image_path,
        "carrier_only_counterfactual_image_digest": carrier_image_digest,
    }
    result_metadata = {
        "final_image_attention_observability": accepted,
        "carrier_only_final_image_preservation": preservation,
    }
    assert _final_image_attention_observability_ready(
        {"metadata": result_metadata},
        config,
    ) is False
    gpu_record = dict(accepted)
    gpu_record[
        "final_image_attention_observability_gpu_execution_verified"
    ] = True
    assert _final_image_attention_observability_ready(
        {
            "metadata": {
                **result_metadata,
                "final_image_attention_observability": gpu_record,
            }
        },
        config,
    ) is True
    mismatched_preservation = dict(preservation)
    mismatched_preservation["carrier_only_counterfactual_identity_digest"] = (
        "f" * 64
    )
    assert _final_image_attention_observability_ready(
        {
            "metadata": {
                "final_image_attention_observability": gpu_record,
                "carrier_only_final_image_preservation": (
                    mismatched_preservation
                ),
            }
        },
        config,
    ) is False


@pytest.mark.quick
def test_attention_gate_rejects_blind_only_gain_without_frozen_pair_gain() -> None:
    """盲选择增益过线但冻结 carrier pair 不过线时必须拒绝。"""

    threshold = 0.0001

    assert _final_image_attention_attribution_gate_ready(
        blind_attribution_gain=0.2,
        frozen_pair_attribution_gain=threshold,
        minimum_gain=threshold,
        measured_values=(0.2, threshold),
        relation_identity_ready=True,
    ) is False
    assert _final_image_attention_attribution_gate_ready(
        blind_attribution_gain=0.2,
        frozen_pair_attribution_gain=0.2,
        minimum_gain=threshold,
        measured_values=(0.2, 0.2),
        relation_identity_ready=True,
    ) is True


@pytest.mark.quick
def test_final_image_attention_gate_fails_closed_without_qk_extractor() -> None:
    """正式注意力分支缺少最终成图 Q/K 提取器时不得回退到 latent 分数。"""

    with pytest.raises(RuntimeError, match="缺少真实 Q/K 提取器"):
        _final_image_attention_observability_record(
            None,
            object(),
            object(),
            object(),
            SemanticWatermarkRuntimeConfig(),
            carrier_only_counterfactual={
                "carrier_only_counterfactual_ready": True
            },
            require_gpu_execution=False,
        )


@pytest.mark.quick
@pytest.mark.parametrize("invalid_threshold", (0.0, float("nan")))
def test_final_image_attention_gain_threshold_must_be_positive(
    invalid_threshold: float,
) -> None:
    """可观测性下界必须为严格正有限数。"""

    with pytest.raises(ValueError, match="正有限数"):
        SemanticWatermarkRuntimeConfig(
            minimum_final_image_attention_score_gain=invalid_threshold,
        )


@pytest.mark.quick
def test_carrier_only_counterfactual_binds_same_seed_and_scheduler() -> None:
    """反事实必须绑定首 latent、完整调度与无 attention 更新原子。"""

    full_config = SemanticWatermarkRuntimeConfig()
    carrier_config = replace(
        full_config,
        attention_geometry_enabled=False,
    )

    full_records = _counterfactual_update_records(
        full_config,
        role="full_method",
        attention_enabled=True,
    )
    carrier_records = _counterfactual_update_records(
        full_config,
        role="carrier_only_counterfactual",
        attention_enabled=False,
    )
    identity = _carrier_only_counterfactual_identity(
        full_config,
        carrier_config,
        full_records,
        carrier_records,
    )

    assert identity["carrier_only_counterfactual_ready"] is True
    assert identity["carrier_only_counterfactual_changed_fields"] == [
        "attention_geometry_enabled"
    ]
    assert len(identity["carrier_only_counterfactual_identity_digest"]) == 64
    assert identity["full_method_initial_latent_content_sha256"] == (
        identity["carrier_only_initial_latent_content_sha256"]
    )
    assert identity["carrier_only_counterfactual_update_count"] == len(
        full_config.injection_step_indices
    )
    assert identity["carrier_only_counterfactual_effect_scope"] == (
        "attention_geometry_switch_total_mechanism_effect"
    )
    assert identity[
        "carrier_only_counterfactual_realized_carrier_equality_assumed"
    ] is False

    drifted_records = _counterfactual_update_records(
        full_config,
        role="carrier_only_counterfactual",
        attention_enabled=False,
    )
    drifted_records[0]["timestep"] = -1.0
    with pytest.raises(RuntimeError, match="scheduler 轨迹"):
        _carrier_only_counterfactual_identity(
            full_config,
            carrier_config,
            full_records,
            drifted_records,
        )

    initial_latent_drift = _counterfactual_update_records(
        full_config,
        role="carrier_only_counterfactual",
        attention_enabled=False,
    )
    initial_latent_drift[0]["latent_content_sha256_before"] = "b" * 64
    with pytest.raises(RuntimeError, match="首个注入前 latent"):
        _carrier_only_counterfactual_identity(
            full_config,
            carrier_config,
            full_records,
            initial_latent_drift,
        )

    attention_atom_drift = _counterfactual_update_records(
        full_config,
        role="carrier_only_counterfactual",
        attention_enabled=False,
    )
    attention_atom_drift[0]["attention_score_before"] = 0.1
    with pytest.raises(RuntimeError, match="仍包含 attention 数值"):
        _carrier_only_counterfactual_identity(
            full_config,
            carrier_config,
            full_records,
            attention_atom_drift,
        )

    source_drift = _counterfactual_update_records(
        full_config,
        role="carrier_only_counterfactual",
        attention_enabled=False,
    )
    source_drift[0]["metadata"]["attention_source"] = "real_qk_projection"
    with pytest.raises(RuntimeError, match="错误声明真实 Q/K"):
        _carrier_only_counterfactual_identity(
            full_config,
            carrier_config,
            full_records,
            source_drift,
        )

    pair_identity_drift = _counterfactual_update_records(
        full_config,
        role="carrier_only_counterfactual",
        attention_enabled=False,
    )
    pair_identity_drift[0]["stable_pair_weight_identity_digest"] = "c" * 64
    with pytest.raises(RuntimeError, match="仍包含 attention 或 pair 身份"):
        _carrier_only_counterfactual_identity(
            full_config,
            carrier_config,
            full_records,
            pair_identity_drift,
        )

    direct_source_drift = _counterfactual_update_records(
        full_config,
        role="carrier_only_counterfactual",
        attention_enabled=False,
    )
    direct_source_drift[0]["attention_relation_direct_qk_source_ready"] = True
    with pytest.raises(RuntimeError, match="错误声明直接 Q/K"):
        _carrier_only_counterfactual_identity(
            full_config,
            carrier_config,
            full_records,
            direct_source_drift,
        )

    null_space_drift = _counterfactual_update_records(
        full_config,
        role="carrier_only_counterfactual",
        attention_enabled=False,
    )
    null_space_drift[0]["null_space_records"]["attention_geometry"] = {}
    with pytest.raises(RuntimeError, match="活动分支身份"):
        _carrier_only_counterfactual_identity(
            full_config,
            carrier_config,
            full_records,
            null_space_drift,
        )


@pytest.mark.quick
def test_carrier_only_artifact_binding_rejects_manifest_or_file_drift(
    tmp_path,
) -> None:
    """反事实原子、图像、记录与 manifest 必须可重建为同一身份。"""

    config = SemanticWatermarkRuntimeConfig(attention_geometry_enabled=True)
    carrier_config = replace(config, attention_geometry_enabled=False)
    full_records = _counterfactual_update_records(
        config,
        role="full_method",
        attention_enabled=True,
    )
    carrier_records = _counterfactual_update_records(
        config,
        role="carrier_only_counterfactual",
        attention_enabled=False,
    )
    identity = _carrier_only_counterfactual_identity(
        config,
        carrier_config,
        full_records,
        carrier_records,
    )
    carrier_path = tmp_path / "outputs" / "runtime" / "carrier_only_image.png"
    carrier_path.parent.mkdir(parents=True)
    carrier_path.write_bytes(b"carrier-only-image")
    full_record_path = carrier_path.parent / "latent_update_records.jsonl"
    carrier_atom_path = carrier_path.parent / "carrier_only_update_records.jsonl"

    def write_records(path, records) -> None:
        """按正式 JSONL 序列化规则写入更新原子。"""

        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in records
            ),
            encoding="utf-8",
        )

    write_records(full_record_path, full_records)
    write_records(carrier_atom_path, carrier_records)
    image_relative_path = carrier_path.relative_to(tmp_path).as_posix()
    full_relative_path = full_record_path.relative_to(tmp_path).as_posix()
    atom_relative_path = carrier_atom_path.relative_to(tmp_path).as_posix()
    image_digest = file_digest(carrier_path)
    shared_identity = {
        **identity,
        "carrier_only_counterfactual_image_path": image_relative_path,
        "carrier_only_counterfactual_image_digest": image_digest,
        "carrier_only_counterfactual_atom_path": atom_relative_path,
        "carrier_only_counterfactual_atom_file_sha256": file_digest(
            carrier_atom_path
        ),
    }
    result = {
        "update_record_path": full_relative_path,
        "metadata": {
            "final_image_attention_observability": dict(shared_identity),
            "carrier_only_final_image_preservation": {
                **shared_identity,
                "carrier_only_final_image_preservation_gate_ready": True,
                "carrier_only_to_full_final_image_preservation_gate_ready": True,
                "carrier_only_counterfactual_three_way_preservation_gate_ready": True,
            },
            "final_image_preservation": {
                "final_image_preservation_gate_ready": True,
            },
            "carrier_only_counterfactual": dict(shared_identity),
        }
    }
    manifest = {
        "output_paths": [
            image_relative_path,
            full_relative_path,
            atom_relative_path,
        ],
        "metadata": {
            "carrier_only_counterfactual_identity_digest": identity[
                "carrier_only_counterfactual_identity_digest"
            ],
            "carrier_only_counterfactual_image_digest": image_digest,
            "carrier_only_counterfactual_atom_path": atom_relative_path,
            "carrier_only_counterfactual_atom_file_sha256": file_digest(
                carrier_atom_path
            ),
            "carrier_only_counterfactual_atom_content_digest": identity[
                "carrier_only_counterfactual_atom_content_digest"
            ],
        },
    }

    assert _carrier_only_counterfactual_artifact_binding_ready(
        result,
        manifest,
        tmp_path.resolve(),
        config,
    ) is True
    drifted_manifest = {
        **manifest,
        "metadata": {
            **manifest["metadata"],
            "carrier_only_counterfactual_identity_digest": "b" * 64,
        },
    }
    assert _carrier_only_counterfactual_artifact_binding_ready(
        result,
        drifted_manifest,
        tmp_path.resolve(),
        config,
    ) is False

    carrier_atom_path.write_text(
        carrier_atom_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert _carrier_only_counterfactual_artifact_binding_ready(
        result,
        manifest,
        tmp_path.resolve(),
        config,
    ) is False

    tampered_records = [dict(record) for record in carrier_records]
    tampered_records[0]["attention_score_before"] = 0.1
    write_records(carrier_atom_path, tampered_records)
    tampered_file_sha256 = file_digest(carrier_atom_path)
    tampered_result = {
        **result,
        "metadata": {
            **result["metadata"],
            "final_image_attention_observability": {
                **result["metadata"]["final_image_attention_observability"],
                "carrier_only_counterfactual_atom_file_sha256": (
                    tampered_file_sha256
                ),
            },
            "carrier_only_final_image_preservation": {
                **result["metadata"]["carrier_only_final_image_preservation"],
                "carrier_only_counterfactual_atom_file_sha256": (
                    tampered_file_sha256
                ),
            },
            "carrier_only_counterfactual": {
                **result["metadata"]["carrier_only_counterfactual"],
                "carrier_only_counterfactual_atom_file_sha256": (
                    tampered_file_sha256
                ),
            },
        },
    }
    tampered_manifest = {
        **manifest,
        "metadata": {
            **manifest["metadata"],
            "carrier_only_counterfactual_atom_file_sha256": (
                tampered_file_sha256
            ),
        },
    }
    assert _carrier_only_counterfactual_artifact_binding_ready(
        tampered_result,
        tampered_manifest,
        tmp_path.resolve(),
        config,
    ) is False
