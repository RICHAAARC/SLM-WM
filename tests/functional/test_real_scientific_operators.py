"""验证真实风险、Jacobian、载体、注意力和仅图像检测算子。"""

from __future__ import annotations

import inspect
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch

from main.methods.carrier import (
    build_low_frequency_template,
    build_tail_robust_template,
    compute_blind_content_score,
)
from main.methods.detection import ImageOnlyDetectionConfig, detect_image_only_watermark
from main.methods.geometry import (
    DifferentiableAttentionRecorder,
    attention_geometry_score,
    attention_relation_stability_map,
    optimize_attention_geometry_update,
    qk_self_attention,
    recover_attention_affine_alignment,
)
from main.methods.geometry.differentiable_attention import keyed_relation_signs
from main.methods.semantic import build_branch_risk_fields
from main.methods.subspace import (
    JacobianNullSpaceResult,
    build_exact_jvp_linearization,
    exact_jvp,
    generate_keyed_candidate_directions,
    solve_jacobian_null_space,
)
from experiments.runners.image_only_dataset_runtime import (
    FrozenEvidenceProtocol,
    _scientific_update_record_ready,
    apply_frozen_evidence_protocol,
    calibrate_complete_evidence_protocol,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    _image_attention_extractor,
    _public_detection_noise_seed,
    build_semantic_watermark_run_id,
    load_completed_semantic_watermark_runtime_result,
    semantic_watermark_runtime_config_payload,
)
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from scripts import semantic_watermark_scientific_workflow as scientific_workflow
from tests.helpers.scientific_unit_provenance import (
    build_test_scientific_unit_provenance,
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
        stability_values=(0.8, 0.8),
        saliency_values=(0.2, 0.2),
        attention_stability_values=(0.8, 0.8),
    )

    assert fields.lf_content.risk_values[0] < fields.lf_content.risk_values[1]
    assert fields.tail_robust.risk_values[0] > fields.tail_robust.risk_values[1]
    assert fields.lf_content.risk_field_digest != fields.tail_robust.risk_field_digest


@pytest.mark.quick
def test_exact_jvp_and_svd_recover_zero_response_latent_direction() -> None:
    """真实 JVP 与 SVD 应恢复语义和视觉特征都不响应的 latent 方向。"""

    latent = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    def semantic(values: torch.Tensor) -> torch.Tensor:
        return torch.stack((values[0] ** 2, 3.0 * values[1]))

    def visual(values: torch.Tensor) -> torch.Tensor:
        return torch.stack((2.0 * values[2],))

    _, tangent = exact_jvp(semantic, latent, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    result = solve_jacobian_null_space(
        latent=latent,
        semantic_feature_function=semantic,
        visual_feature_function=visual,
        candidate_matrix=torch.eye(4),
        null_rank=1,
        branch_name="lf_content",
    )

    assert tangent.tolist() == pytest.approx([2.0, 0.0])
    assert result.response_residual == pytest.approx(0.0, abs=1e-7)
    assert result.orthogonality_error == pytest.approx(0.0, abs=1e-6)
    assert abs(float(result.latent_basis[3, 0])) == pytest.approx(1.0, abs=1e-6)
    assert result.metadata["jvp_mode"] == "torch_autograd_exact_jvp"


@pytest.mark.quick
def test_reusable_exact_jvp_linearization_preserves_null_space_solution() -> None:
    """共享线性算子必须保持真实 JVP 与低响应方向, 不能退化为数值差分。"""

    latent = torch.tensor([1.0, 2.0, 3.0, 4.0])

    def joint(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.stack((values[0] ** 2, 3.0 * values[1])), torch.stack((2.0 * values[2],))

    linearization = build_exact_jvp_linearization(joint, latent)
    result = solve_jacobian_null_space(
        latent=latent,
        semantic_feature_function=lambda values: joint(values)[0],
        visual_feature_function=lambda values: joint(values)[1],
        candidate_matrix=torch.eye(4),
        null_rank=1,
        branch_name="lf_content",
        joint_feature_function=joint,
        joint_feature_linearization=linearization,
    )

    assert result.response_residual == pytest.approx(0.0, abs=1e-7)
    assert abs(float(result.latent_basis[3, 0])) == pytest.approx(1.0, abs=1e-6)
    assert "exact_jvp" in result.metadata["jvp_mode"]


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
def test_semantic_condition_solver_returns_algebraic_null_space() -> None:
    """20个候选与16个独立条件必须产生近零响应的4维 Null Space。"""

    latent = torch.zeros(1, 1, 5, 5)
    candidates = torch.eye(25)[:, :20]

    result = solve_jacobian_null_space(
        latent=latent,
        semantic_feature_function=lambda value: value.reshape(-1)[:8],
        visual_feature_function=lambda value: value.reshape(-1)[8:16],
        candidate_matrix=candidates,
        null_rank=4,
        branch_name="lf_content",
    )

    assert result.basis_rank == 4
    assert result.relative_response_residual <= 1e-6
    assert result.orthogonality_error <= 1e-6


@pytest.mark.quick
def test_scientific_operator_gate_requires_all_real_operator_evidence() -> None:
    """关键算子门禁必须同时检查 JVP、残差、载体能量和 Q/K 提升。"""

    subspace = {
        "response_residual": 0.1,
        "relative_response_residual": 1e-6,
        "orthogonality_error": 1e-6,
        "metadata": {
            "jvp_mode": "torch_func_linearize_exact_jvp",
            "preferred_direction_count": 1,
        },
    }
    record = {
        "branch_risk_bundle_digest": "risk_digest",
        "branch_risk_records": {
            name: {"eligible_position_count": 10}
            for name in ("lf_content", "tail_robust", "attention_geometry")
        },
        "null_space_records": {
            "lf_content": dict(subspace),
            "tail_robust": dict(subspace),
            "attention_geometry": dict(subspace),
        },
        "lf_projection_energy_retention": 0.2,
        "tail_projection_energy_retention": 0.2,
        "attention_score_gain": 0.01,
        "attention_applied_update_strength": 0.001,
    }
    config = SemanticWatermarkRuntimeConfig()

    assert _scientific_update_record_ready(record, config) is True
    record["attention_score_gain"] = 0.0
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
def test_qk_sampling_preserves_two_dimensional_token_grid() -> None:
    """有界 Q/K 抽样必须沿二维行列轴取点, 不能等距抽一维序号。"""

    module = _ToyAttention(4)
    hidden_states = torch.randn(1, 16, 4)

    _, indices = qk_self_attention(module, hidden_states, max_tokens=4)

    assert indices == (0, 3, 12, 15)


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


def _identity_null_space(latent: torch.Tensor) -> JacobianNullSpaceResult:
    """构造完整空间基底, 隔离注意力梯度测试。"""

    element_count = latent.numel()
    identity = torch.eye(element_count)
    return JacobianNullSpaceResult(
        branch_name="attention_geometry",
        candidate_matrix=identity,
        response_matrix=torch.zeros(1, element_count),
        coefficient_basis=identity,
        latent_basis=identity,
        singular_values=(0.0,) * element_count,
        selected_response_values=(0.0,) * element_count,
        response_residual=0.0,
        relative_response_residual=0.0,
        orthogonality_error=0.0,
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

    with DifferentiableAttentionRecorder((('toy_attention', module),), max_tokens=4) as recorder:
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
    with DifferentiableAttentionRecorder((("toy_attention", module),), max_tokens=4) as recorder:
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
    canonical = torch.softmax(2.0 * relation_signs, dim=-1).unsqueeze(0)
    index = torch.tensor(permutation, dtype=torch.long)
    observed = canonical.index_select(1, index).index_select(2, index)

    result = recover_attention_affine_alignment(
        observed,
        key_material,
        layer_name,
        tuple(range(token_count)),
    )

    assert transform_name
    assert result.geometry_reliable is True
    assert result.inlier_ratio == pytest.approx(1.0)
    assert result.relation_sync_score > 0.95
    assert result.metadata["matcher"] == "double_sided_keyed_relation_graph_registration"


@pytest.mark.quick
def test_image_only_detector_reextracts_qk_after_alignment() -> None:
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
    canonical_attention = torch.softmax(2.0 * relation_signs, dim=-1).unsqueeze(0)
    flip = torch.tensor(
        [row * 8 + (7 - column) for row in range(8) for column in range(8)],
        dtype=torch.long,
    )
    observed_attention = canonical_attention.index_select(1, flip).index_select(2, flip)
    reference = torch.zeros(1, 2, 8, 8)
    lf_template = build_low_frequency_template(reference, key_material, model_id)
    tail_template = build_tail_robust_template(reference, key_material, model_id, 0.20)[0]
    original = {"latent": torch.zeros_like(reference), "attention": observed_attention}
    aligned = {
        "latent": 0.8 * lf_template + 0.4 * tail_template,
        "attention": canonical_attention,
    }
    extraction_count = 0

    def extract(sample: dict[str, torch.Tensor]) -> tuple[tuple[str, torch.Tensor, tuple[int, ...]], ...]:
        nonlocal extraction_count
        extraction_count += 1
        return ((layer_name, sample["attention"], tuple(range(token_count))),)

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
        ),
        image_latent_encoder=lambda sample: sample["latent"],
        image_attention_extractor=extract,
        image_aligner=lambda _image, _alignment: aligned,
    )

    assert extraction_count == 2
    assert result.raw_attention_geometry_score is not None
    assert result.attention_geometry_score is not None
    assert result.attention_geometry_score > 0.95
    assert result.attention_sync_score is not None and result.attention_sync_score > 0.95
    assert result.geometry_reliable is True
    assert result.rescue_applied is True


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
def test_post_step_injection_rejects_last_scheduler_step() -> None:
    """callback-on-step-end 的最后一步没有下一时刻, 必须在配置层拒绝。"""

    base = SemanticWatermarkRuntimeConfig()
    with pytest.raises(ValueError, match="post-step"):
        replace(base, injection_step_indices=(base.inference_steps - 1,))


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

    config = SemanticWatermarkRuntimeConfig(output_dir="outputs/cache_test")
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
