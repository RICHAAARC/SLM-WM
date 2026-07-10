"""验证真实风险、Jacobian、载体、注意力和仅图像检测算子。"""

from __future__ import annotations

import inspect
from dataclasses import asdict, replace
import json
from pathlib import Path

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
    attention_relation_stability_map,
    optimize_attention_geometry_update,
    qk_self_attention,
)
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
    _public_detection_noise_seed,
    build_semantic_watermark_run_id,
    load_completed_semantic_watermark_runtime_result,
)
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from paper_workflow.colab_utils import semantic_watermark_image_only as colab_image_only


@pytest.mark.quick
def test_image_only_attention_noise_seed_does_not_depend_on_generation_seed_or_prompt() -> None:
    """盲检公开噪声不得依赖生成种子、Prompt 或样本序号。"""

    base = SemanticWatermarkRuntimeConfig()
    changed_sample = replace(base, seed=base.seed + 999, prompt="完全不同的生成条件", prompt_id="other")

    assert _public_detection_noise_seed(base) == _public_detection_noise_seed(changed_sample)
    assert _public_detection_noise_seed(base) != _public_detection_noise_seed(
        replace(base, model_id="different-public-model")
    )


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


@pytest.mark.quick
def test_frozen_protocol_recomputes_threshold_dependent_failure_reason() -> None:
    """冻结阈值改变后必须重算失败原因, 不能沿用预检测阈值的分类。"""

    protocol = FrozenEvidenceProtocol(
        content_threshold=0.5,
        rescue_margin_low=-0.2,
        geometry_score_threshold=0.0,
        geometry_calibration_negative_count=10,
        geometry_calibration_exceedance_count=0,
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
        "geometry_reliable": True,
        "alignment": {"geometry_reliable": True},
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
    result_payload = {
        "run_id": run_id,
        "run_decision": "pass",
        **{key: path.relative_to(tmp_path).as_posix() for key, path in files.items()},
        "manifest_path": manifest_path.relative_to(tmp_path).as_posix(),
        "update_count": 1,
        "clean_detection_positive": False,
        "watermarked_detection_positive": True,
        "elapsed_seconds": 1.0,
        "metadata": {},
    }
    result_path.write_text(json.dumps(result_payload), encoding="utf-8")
    config_payload = {**asdict(config), "key_material": build_stable_digest({"key_material": config.key_material})}
    output_paths = [path.relative_to(tmp_path).as_posix() for path in files.values()]
    output_paths.extend((result_path.relative_to(tmp_path).as_posix(), manifest_path.relative_to(tmp_path).as_posix()))
    manifest_path.write_text(
        json.dumps(
            {
                "config_digest": build_stable_digest(config_payload),
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
    monkeypatch.setattr(colab_image_only, "_run_repository_script", lambda root, script: None)

    summary = colab_image_only.run_semantic_watermark_image_only_session(tmp_path)

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
    monkeypatch.setattr(colab_image_only, "_run_repository_script", lambda root, script: None)

    summary = colab_image_only.run_semantic_watermark_image_only_session(tmp_path)

    assert summary["workflow_decision"] == "dataset_complete"
    assert (runtime_drive_dir / "image_only_dataset_runtime_package_fixture.zip").read_bytes() == b"runtime"
    assert (quality_drive_dir / "dataset_level_quality_package_fixture.zip").read_bytes() == b"quality"
