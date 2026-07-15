"""验证模型资源登记与主方法精确 revision 传递."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.protocol.method_runtime_config import (
    formal_method_config_digest,
    formal_method_config_payload,
    load_formal_method_runtime_config,
    resolve_formal_method_config_path,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    _build_image_only_measurement_config,
    semantic_watermark_runtime_config_payload,
)
from experiments.runtime.diffusion import sd3_pipeline_runtime, semantic_model_loader
from experiments.runtime.model_sources import (
    MODEL_SOURCE_REGISTRY_PATH,
    get_model_source,
    load_model_source_registry,
    require_registered_model_reference,
)
from experiments.runners import image_only_dataset_workload
from main.methods.carrier import LowFrequencyCarrierConfig


@pytest.mark.quick
def test_primary_model_config_matches_immutable_source_registry() -> None:
    """主方法配置必须与登记表中的 SD3.5 和 CLIP 精确提交一致."""

    config = SemanticWatermarkRuntimeConfig()
    diffusion_source = get_model_source("stabilityai_stable_diffusion_3_5_medium")
    vision_source = get_model_source("openai_clip_vit_base_patch32")
    method_config = load_formal_method_runtime_config(".")

    assert (config.model_id, config.model_revision) == (
        diffusion_source.repository_id,
        diffusion_source.revision,
    )
    assert (config.vision_model_id, config.vision_model_revision) == (
        vision_source.repository_id,
        vision_source.revision,
    )
    assert method_config.model_revision == diffusion_source.revision
    assert method_config.vision_model_revision == vision_source.revision
    assert (
        method_config.pipeline_class_name,
        method_config.vae_class_name,
        method_config.transformer_class_name,
        method_config.scheduler_class_name,
    ) == (
        "diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline",
        "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
        "diffusers.models.transformers.transformer_sd3.SD3Transformer2DModel",
        "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
    )
    assert (
        method_config.vae_scaling_factor,
        method_config.vae_shift_factor,
        method_config.latent_torch_dtype,
        method_config.vision_torch_dtype,
    ) == (1.5305, 0.0609, "float16", "float32")
    assert method_config.public_detection_schedule_index == 7
    assert method_config.public_detection_schedule_index == (
        method_config.injection_step_indices[0] + 1
    )
    assert method_config.public_detection_noise_prg_protocol == (
        "sha256_counter_normal_icdf_table20_float32"
    )
    assert method_config.public_detection_noise_domain == (
        "public_image_only_qk_detection_noise"
    )
    assert method_config.public_detection_conditioning_protocol == (
        "sd3_empty_text_triplet_without_cfg"
    )
    assert method_config.public_detection_condition_text == ""
    assert (
        method_config.attention_anchor_count,
        method_config.attention_residual_threshold,
        method_config.attention_minimum_inlier_ratio,
    ) == (12, 0.20, 0.50)
    assert method_config.formal_method_config_digest == (
        formal_method_config_digest(method_config)
    )
    assert config.formal_method_config_digest == (
        method_config.formal_method_config_digest
    )
    formal_settings = method_config.paper_method_settings()
    runtime_payload = semantic_watermark_runtime_config_payload(config)
    assert set(formal_settings).issubset(
        SemanticWatermarkRuntimeConfig.__dataclass_fields__
    )
    assert set(formal_settings).issubset(runtime_payload)
    for field_name, expected_value in formal_settings.items():
        runtime_value = getattr(config, field_name)
        if field_name.endswith("_risk_config"):
            runtime_value = asdict(runtime_value)
        assert runtime_value == expected_value
    assert config.inference_steps == method_config.inference_steps
    assert config.injection_step_indices == method_config.injection_step_indices
    assert config.candidate_count == method_config.jacobian_candidate_count
    assert config.null_rank == method_config.null_space_rank
    assert config.attention_module_names == (
        "transformer_blocks.0.attn",
        "transformer_blocks.23.attn",
    )
    assert config.attention_module_names == method_config.attention_module_names
    assert config.attention_coordinate_convention == (
        "normalized_xy_token_centers_corner_endpoints"
    )
    assert config.attention_grid_align_corners is True
    assert method_config.risk_signal_calibration_protocol == (
        "analytic_bounded_branch_signals"
    )
    assert (
        method_config.risk_image_signal_interpolation_mode,
        method_config.risk_image_signal_align_corners,
        method_config.risk_attention_signal_interpolation_mode,
        method_config.risk_attention_signal_align_corners,
    ) == ("bilinear", False, "bilinear", True)
    assert method_config.risk_neutral_texture_value == 0.5
    assert method_config.risk_eligibility_comparison == "strict_less_than"
    assert method_config.risk_budget_broadcast_protocol == (
        "per_sample_hw_repeat_channels_nchw"
    )
    assert method_config.risk_zero_support_protocol == (
        "exact_zero_direction_or_fail_closed"
    )
    assert method_config.risk_bounded_scale_protocol == (
        "direction_peak_frozen_budget_ceiling_box"
    )
    assert method_config.risk_bounded_scale_direction_epsilon == 1e-12
    assert asdict(method_config.lf_content_risk_config) == {
        "local_contrast_risk_weight": 0.30,
        "semantic_weight": 0.30,
        "texture_weight": 0.20,
        "adjacent_step_instability_weight": 0.20,
        "attention_instability_weight": 0.0,
        "texture_preference": "avoid",
        "eligibility_threshold": 0.55,
        "budget_floor": 0.05,
        "budget_ceiling": 1.0,
        "budget_gain": 0.70,
    }
    assert asdict(method_config.tail_robust_risk_config) == {
        "local_contrast_risk_weight": 0.25,
        "semantic_weight": 0.25,
        "texture_weight": 0.30,
        "adjacent_step_instability_weight": 0.20,
        "attention_instability_weight": 0.0,
        "texture_preference": "prefer",
        "eligibility_threshold": 0.55,
        "budget_floor": 0.05,
        "budget_ceiling": 1.0,
        "budget_gain": 0.70,
    }
    assert asdict(method_config.attention_geometry_risk_config) == {
        "local_contrast_risk_weight": 0.20,
        "semantic_weight": 0.25,
        "texture_weight": 0.05,
        "adjacent_step_instability_weight": 0.20,
        "attention_instability_weight": 0.30,
        "texture_preference": "neutral",
        "eligibility_threshold": 0.55,
        "budget_floor": 0.05,
        "budget_ceiling": 1.0,
        "budget_gain": 0.70,
    }
    assert (
        method_config.null_space_numerical_epsilon,
        method_config.maximum_qr_condition_number,
        method_config.maximum_orthogonality_error,
        method_config.qr_reference_solve_protocol,
    ) == (
        1e-12,
        1e6,
        1e-5,
        "right_upper_triangular_solve_without_explicit_inverse",
    )
    assert (
        method_config.lf_kernel_size,
        method_config.lf_stride,
        method_config.lf_padding,
        method_config.lf_boundary_mode,
        method_config.lf_ceil_mode,
        method_config.lf_count_include_pad,
        method_config.lf_divisor_override,
        method_config.lf_detection_score_weight,
        method_config.tail_robust_detection_score_weight,
    ) == (5, 1, 2, "zero_padding", False, True, None, 0.70, 0.30)
    assert (
        method_config.quantized_branch_composition_protocol,
        method_config.quantized_branch_composition_order,
        method_config.combined_budget_envelope_rule,
        method_config.quantized_budget_envelope_absolute_tolerance,
        method_config.quantized_budget_envelope_backtracking_factor,
        method_config.quantized_budget_envelope_backtracking_maximum_steps,
    ) == (
        "float32_ordered_branch_sum_add_float32_latent_single_cast",
        ("lf_content", "tail_robust", "attention_geometry"),
        "sum_active_branch_envelopes",
        0.0,
        0.5,
        24,
    )
    assert diffusion_source.revision_url.endswith(f"/tree/{diffusion_source.revision}")
    assert "primary_diffusion_model" in diffusion_source.usage_roles
    assert "semantic_condition_encoder" in vision_source.usage_roles
    assert config.carrier_model_reference == (
        "stabilityai/stable-diffusion-3.5-medium@"
        "b940f670f0eda2d07fbb75229e779da1ad11eb80"
    )
    with pytest.raises(ValueError, match="attention_module_names"):
        SemanticWatermarkRuntimeConfig(
            attention_module_names=(
                "transformer_blocks.1.attn",
                "transformer_blocks.22.attn",
            )
        )


@pytest.mark.quick
def test_dataset_runtime_entrypoint_uses_registered_revision_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """独立脚本未显式覆盖模型时必须沿用登记表中的精确提交."""

    for variable_name in (
        "SLM_WM_MODEL_ID",
        "SLM_WM_MODEL_REVISION",
        "SLM_WM_VISION_MODEL_ID",
        "SLM_WM_VISION_MODEL_REVISION",
    ):
        monkeypatch.delenv(variable_name, raising=False)

    config = image_only_dataset_workload.build_method_config(".")

    assert config.model_revision == get_model_source(
        "stabilityai_stable_diffusion_3_5_medium"
    ).revision
    assert config.vision_model_revision == get_model_source(
        "openai_clip_vit_base_patch32"
    ).revision
    assert config.inference_steps == load_formal_method_runtime_config(".").inference_steps


@pytest.mark.quick
def test_dataset_runtime_rejects_method_environment_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式脚本不得通过环境变量形成未登记的模型或方法参数分支。"""

    monkeypatch.setenv("SLM_WM_MODEL_REVISION", "0" * 40)

    with pytest.raises(ValueError, match="model_sd35.yaml 不一致"):
        image_only_dataset_workload.build_method_config(".")


@pytest.mark.quick
def test_dataset_runtime_rejects_risk_protocol_environment_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """环境变量不得把严格风险资格边界改为未登记比较规则。"""

    monkeypatch.setenv(
        "SLM_WM_RISK_ELIGIBILITY_COMPARISON",
        "less_than_or_equal",
    )

    with pytest.raises(ValueError, match="model_sd35.yaml 不一致"):
        image_only_dataset_workload.build_method_config(".")


@pytest.mark.quick
def test_dataset_runtime_rejects_alignment_gate_environment_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """环境变量不得改写预注册注意力结构门禁."""

    monkeypatch.setenv("SLM_WM_ATTENTION_ANCHOR_COUNT", "13")

    with pytest.raises(ValueError, match="model_sd35.yaml 不一致"):
        image_only_dataset_workload.build_method_config(".")


@pytest.mark.quick
def test_paper_method_settings_include_frozen_risk_and_write_protocols() -> None:
    """三级论文配置必须共享完整风险、Null Space 和量化合成常量。"""

    settings = load_formal_method_runtime_config(".").paper_method_settings()

    assert settings["risk_eligibility_comparison"] == "strict_less_than"
    assert settings["risk_neutral_texture_value"] == 0.5
    assert settings["risk_budget_broadcast_protocol"] == (
        "per_sample_hw_repeat_channels_nchw"
    )
    assert settings["lf_content_risk_config"]["budget_ceiling"] == 1.0
    assert settings["qr_reference_solve_protocol"] == (
        "right_upper_triangular_solve_without_explicit_inverse"
    )
    assert settings["quantized_branch_composition_protocol"] == (
        "float32_ordered_branch_sum_add_float32_latent_single_cast"
    )
    assert settings["quantized_budget_envelope_backtracking_maximum_steps"] == 24
    assert settings["attention_anchor_count"] == 12
    assert settings["attention_residual_threshold"] == 0.20
    assert settings["attention_minimum_inlier_ratio"] == 0.50
    assert (
        settings["lf_kernel_size"],
        settings["lf_stride"],
        settings["lf_padding"],
        settings["lf_boundary_mode"],
        settings["lf_ceil_mode"],
        settings["lf_count_include_pad"],
        settings["lf_divisor_override"],
    ) == (5, 1, 2, "zero_padding", False, True, None)
    assert settings["lf_detection_score_weight"] == 0.70
    assert settings["tail_robust_detection_score_weight"] == 0.30


@pytest.mark.quick
def test_attention_alignment_gate_is_frozen_in_formal_config_and_digest() -> None:
    """正式配置摘要必须逐字段绑定预注册注意力结构门禁."""

    config = load_formal_method_runtime_config(".")
    payload = formal_method_config_payload(config)

    def independent_digest(value: object) -> str:
        """以独立规范 JSON 公式重算正式配置摘要."""

        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    baseline_digest = independent_digest(payload)
    assert baseline_digest == formal_method_config_digest(config)
    for field_name, changed_value in (
        ("attention_anchor_count", 13),
        ("attention_residual_threshold", 0.21),
        ("attention_minimum_inlier_ratio", 0.51),
    ):
        changed_payload = deepcopy(payload)
        changed_payload["formal_method_config"][field_name] = changed_value
        assert independent_digest(changed_payload) != baseline_digest


@pytest.mark.quick
def test_low_frequency_protocol_is_frozen_in_formal_config_and_digest() -> None:
    """正式配置摘要必须逐字段绑定 LF 离散协议和内容权重."""

    config = load_formal_method_runtime_config(".")
    payload = formal_method_config_payload(config)
    baseline_digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert baseline_digest == formal_method_config_digest(config)
    for field_name, changed_value in (
        ("lf_kernel_size", 7),
        ("lf_stride", 2),
        ("lf_padding", 1),
        ("lf_boundary_mode", "reflect"),
        ("lf_ceil_mode", True),
        ("lf_count_include_pad", False),
        ("lf_divisor_override", 9),
        ("lf_detection_score_weight", 0.69),
        ("tail_robust_detection_score_weight", 0.31),
    ):
        changed_payload = deepcopy(payload)
        changed_payload["formal_method_config"][field_name] = changed_value
        changed_digest = hashlib.sha256(
            json.dumps(
                changed_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        assert changed_digest != baseline_digest


@pytest.mark.quick
@pytest.mark.parametrize(
    ("source_line", "changed_line"),
    (
        ("lf_kernel_size: 5", "lf_kernel_size: 7"),
        ("lf_stride: 1", "lf_stride: 2"),
        ("lf_padding: 2", "lf_padding: 1"),
        ("lf_boundary_mode: zero_padding", "lf_boundary_mode: reflect"),
        ("lf_ceil_mode: false", "lf_ceil_mode: true"),
        ("lf_count_include_pad: true", "lf_count_include_pad: false"),
        ("lf_divisor_override: null", "lf_divisor_override: 9"),
        ("lf_detection_score_weight: 0.70", "lf_detection_score_weight: 0.69"),
        (
            "tail_robust_detection_score_weight: 0.30",
            "tail_robust_detection_score_weight: 0.31",
        ),
    ),
)
def test_formal_method_config_rejects_low_frequency_protocol_drift(
    tmp_path: Path,
    source_line: str,
    changed_line: str,
) -> None:
    """唯一 YAML 的任一 LF 离散字段或检测权重漂移都必须失败关闭."""

    source = Path("configs/model_sd35.yaml").read_text(encoding="utf-8")
    changed = source.replace(source_line, changed_line, 1)
    assert changed != source
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "model_sd35.yaml").write_text(changed, encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        load_formal_method_runtime_config(tmp_path)


@pytest.mark.quick
@pytest.mark.parametrize(
    ("source_line", "changed_line"),
    (
        ("attention_anchor_count: 12", "attention_anchor_count: 13"),
        (
            "attention_residual_threshold: 0.20",
            "attention_residual_threshold: 0.21",
        ),
        (
            "attention_minimum_inlier_ratio: 0.50",
            "attention_minimum_inlier_ratio: 0.51",
        ),
    ),
)
def test_formal_method_config_rejects_attention_alignment_gate_drift(
    tmp_path: Path,
    source_line: str,
    changed_line: str,
) -> None:
    """唯一 YAML 中任一注意力结构门禁漂移都必须失败关闭."""

    source = Path("configs/model_sd35.yaml").read_text(encoding="utf-8")
    changed = source.replace(source_line, changed_line, 1)
    assert changed != source
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "model_sd35.yaml").write_text(changed, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="正式注意力配准锚点、残差或内点门禁发生漂移",
    ):
        load_formal_method_runtime_config(tmp_path)


@pytest.mark.quick
def test_runtime_detector_config_consumes_formal_alignment_gate() -> None:
    """运行层必须把唯一正式门禁显式传递给核心盲检器."""

    runtime = SemanticWatermarkRuntimeConfig()
    detector = _build_image_only_measurement_config(runtime)
    payload = semantic_watermark_runtime_config_payload(runtime)

    assert detector.attention_anchor_count == runtime.attention_anchor_count == 12
    assert (
        detector.attention_residual_threshold
        == runtime.attention_residual_threshold
        == 0.20
    )
    assert (
        detector.attention_minimum_inlier_ratio
        == runtime.attention_minimum_inlier_ratio
        == 0.50
    )
    assert payload["attention_anchor_count"] == 12
    assert payload["attention_residual_threshold"] == 0.20
    assert payload["attention_minimum_inlier_ratio"] == 0.50


@pytest.mark.quick
def test_runtime_detector_config_consumes_formal_low_frequency_protocol() -> None:
    """嵌入运行与核心盲检器必须消费同一 LF 对象、权重和 tail 比例."""

    runtime = SemanticWatermarkRuntimeConfig()
    detector = _build_image_only_measurement_config(runtime)
    payload = semantic_watermark_runtime_config_payload(runtime)

    assert isinstance(detector.low_frequency_config, LowFrequencyCarrierConfig)
    assert detector.low_frequency_config == runtime.low_frequency_carrier_config
    assert detector.low_frequency_config.to_record() == (
        runtime.low_frequency_carrier_config.to_record()
    )
    assert detector.lf_weight == runtime.lf_detection_score_weight == 0.70
    assert (
        detector.tail_robust_weight
        == runtime.tail_robust_detection_score_weight
        == 0.30
    )
    assert detector.tail_fraction == runtime.tail_fraction == 0.20
    assert (
        payload["lf_kernel_size"],
        payload["lf_stride"],
        payload["lf_padding"],
        payload["lf_boundary_mode"],
        payload["lf_ceil_mode"],
        payload["lf_count_include_pad"],
        payload["lf_divisor_override"],
    ) == (5, 1, 2, "zero_padding", False, True, None)
    assert payload["lf_detection_score_weight"] == 0.70
    assert payload["tail_robust_detection_score_weight"] == 0.30


@pytest.mark.quick
@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("attention_anchor_count", 12.0),
        ("attention_anchor_count", True),
        ("attention_residual_threshold", float("nan")),
        ("attention_minimum_inlier_ratio", float("inf")),
    ),
)
def test_runtime_config_rejects_invalid_alignment_gate_types(
    field_name: str,
    invalid_value: object,
) -> None:
    """运行配置入口必须先执行精确类型与有限性门禁."""

    with pytest.raises(ValueError):
        replace(
            SemanticWatermarkRuntimeConfig(),
            **{field_name: invalid_value},
        )


@pytest.mark.quick
def test_formal_method_config_rejects_branch_risk_constant_drift(
    tmp_path: Path,
) -> None:
    """YAML 中任一正式风险阈值漂移都必须在配置构造边界失败。"""

    source = Path("configs/model_sd35.yaml").read_text(encoding="utf-8")
    changed = source.replace(
        "  eligibility_threshold: 0.55",
        "  eligibility_threshold: 0.56",
        1,
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "model_sd35.yaml").write_text(changed, encoding="utf-8")

    with pytest.raises(ValueError, match="三分支风险权重、阈值或预算常量发生漂移"):
        load_formal_method_runtime_config(tmp_path)


@pytest.mark.quick
def test_formal_method_config_path_does_not_fall_back_to_package(
    tmp_path: Path,
) -> None:
    """目标根目录缺少唯一 YAML 时不得静默读取当前仓库配置。"""

    with pytest.raises(FileNotFoundError, match="正式方法配置不存在"):
        resolve_formal_method_config_path(tmp_path)
    with pytest.raises(FileNotFoundError, match="正式方法配置不存在"):
        load_formal_method_runtime_config(tmp_path)


@pytest.mark.quick
def test_formal_method_config_digest_is_value_stable(
    tmp_path: Path,
) -> None:
    """YAML 排版不进入摘要, 任何实际配置值变化必须改变摘要。"""

    source = Path("configs/model_sd35.yaml").read_text(encoding="utf-8")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    target = config_dir / "model_sd35.yaml"
    target.write_text(source + "\n# 仅改变 YAML 排版\n", encoding="utf-8")
    baseline = load_formal_method_runtime_config(".")
    reformatted = load_formal_method_runtime_config(tmp_path)

    assert reformatted.formal_method_config_digest == (
        baseline.formal_method_config_digest
    )

    target.write_text(
        source.replace(
            "prompt: a high quality photograph of a glass sphere on a wooden table",
            "prompt: a governed alternate configuration value",
        ),
        encoding="utf-8",
    )
    changed = load_formal_method_runtime_config(tmp_path)
    assert changed.formal_method_config_digest != baseline.formal_method_config_digest


@pytest.mark.quick
def test_model_source_registry_rejects_mutable_or_unregistered_revisions(tmp_path: Path) -> None:
    """登记表和运行引用都不得接受分支名、短提交或未登记提交."""

    payload = json.loads(MODEL_SOURCE_REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["sources"]["openai_clip_vit_base_patch32"]["revision"] = "main"
    invalid_path = tmp_path / "model_source_registry.json"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="40位小写十六进制"):
        load_model_source_registry(invalid_path)
    with pytest.raises(ValueError, match="组合未登记"):
        require_registered_model_reference(
            "openai/clip-vit-base-patch32",
            "0" * 40,
        )
    reference_source = get_model_source("manojb_stable_diffusion_2_1_base")
    with pytest.raises(ValueError, match="用途组合未登记"):
        require_registered_model_reference(
            reference_source.repository_id,
            reference_source.revision,
            required_usage_role="primary_diffusion_model",
        )


@pytest.mark.quick
def test_official_reference_mirror_records_unavailable_upstream_separately() -> None:
    """公开镜像不得被描述为原始上游仓库。"""

    source = get_model_source("manojb_stable_diffusion_2_1_base")

    assert source.repository_id == "Manojb/stable-diffusion-2-1-base"
    assert source.upstream_repository_id == "stabilityai/stable-diffusion-2-1-base"
    assert source.upstream_access_status == "unavailable"


@pytest.mark.quick
def test_openclip_source_registers_exact_checkpoint_file() -> None:
    """官方参考 OpenCLIP 来源必须同时固定 checkpoint 文件摘要与大小."""

    source = get_model_source("laion_clip_vit_g14")

    assert source.repository_id == "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
    assert source.revision == "4b0305adc6802b2632e11cbe6606a9bdd43d35c9"
    assert "official_reference_openclip_encoder" in source.usage_roles
    assert [item.to_dict() for item in source.required_files] == [
        {
            "path": "open_clip_pytorch_model.bin",
            "sha256": "6aac683f899159946bc4ca15228bb7016f3cbb1a2c51f365cba0b23923f344da",
            "size_bytes": 5467006745,
        }
    ]


@pytest.mark.quick
def test_model_source_registry_rejects_invalid_required_file_digest(tmp_path: Path) -> None:
    """文件级登记不得接受可疑路径或非精确 SHA-256."""

    payload = json.loads(MODEL_SOURCE_REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["sources"]["laion_clip_vit_g14"]["required_files"][0]["sha256"] = "invalid"
    invalid_path = tmp_path / "model_source_registry.json"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="64位小写 SHA-256"):
        load_model_source_registry(invalid_path)


@pytest.mark.quick
def test_sd35_pipeline_forwards_registered_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """SD3.5 加载边界必须把精确 revision 传给 from_pretrained."""

    captured: dict[str, object] = {}

    class FakeCuda:
        """提供 CUDA 可用性接口."""

        @staticmethod
        def is_available() -> bool:
            """返回测试所需的可用状态."""

            return True

    class FakeTorch:
        """提供加载器使用的最小 torch 接口."""

        cuda = FakeCuda()
        float16 = "float16"

    class FakeParameter:
        """提供组件 dtype 复验需要的最小参数对象."""

        dtype = "float16"

    class FakeVAE:
        """模拟冻结类身份和 VAE 归一化常量."""

        config = SimpleNamespace(scaling_factor=1.5305, shift_factor=0.0609)

        @staticmethod
        def parameters() -> tuple[FakeParameter, ...]:
            """返回可核验 dtype 的参数."""

            return (FakeParameter(),)

    class FakeTransformer:
        """模拟冻结 SD3 Transformer 类."""

        @staticmethod
        def parameters() -> tuple[FakeParameter, ...]:
            """返回可核验 dtype 的参数."""

            return (FakeParameter(),)

    class FakeScheduler:
        """模拟冻结 FlowMatch scheduler 类."""

    class FakePipeline:
        """记录 from_pretrained 参数而不下载模型."""

        def __init__(self) -> None:
            self.vae = FakeVAE()
            self.transformer = FakeTransformer()
            self.scheduler = FakeScheduler()

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> "FakePipeline":
            """保存模型标识和关键字参数."""

            captured["model_id"] = model_id
            captured.update(kwargs)
            return cls()

        def to(self, device_name: str) -> "FakePipeline":
            """记录目标设备并返回自身."""

            captured["device_name"] = device_name
            return self

        def set_progress_bar_config(self, disable: bool) -> None:
            """记录进度条配置."""

            captured["progress_disabled"] = disable

    FakePipeline.__module__ = (
        "diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3"
    )
    FakePipeline.__qualname__ = "StableDiffusion3Pipeline"
    FakeVAE.__module__ = "diffusers.models.autoencoders.autoencoder_kl"
    FakeVAE.__qualname__ = "AutoencoderKL"
    FakeTransformer.__module__ = (
        "diffusers.models.transformers.transformer_sd3"
    )
    FakeTransformer.__qualname__ = "SD3Transformer2DModel"
    FakeScheduler.__module__ = (
        "diffusers.schedulers.scheduling_flow_match_euler_discrete"
    )
    FakeScheduler.__qualname__ = "FlowMatchEulerDiscreteScheduler"

    monkeypatch.setattr(
        sd3_pipeline_runtime.repository_environment,
        "require_published_formal_execution_lock",
        lambda root: {"lock_digest": "fixture"},
    )
    monkeypatch.setattr(
        sd3_pipeline_runtime,
        "import_runtime_dependencies",
        lambda: (None, FakeTorch, None, FakePipeline),
    )
    monkeypatch.setattr(
        sd3_pipeline_runtime,
        "build_runtime_environment_report",
        lambda profile_id, **kwargs: {
            "dependency_environment_ready": True,
            "dependency_readiness_blockers": [],
        },
    )
    monkeypatch.setattr(sd3_pipeline_runtime, "flatten_environment_versions", lambda report: {})

    config = SemanticWatermarkRuntimeConfig()
    pipeline, runtime_versions = sd3_pipeline_runtime.load_pipeline(config)

    assert captured["model_id"] == config.model_id
    assert captured["revision"] == config.model_revision
    assert captured["torch_dtype"] == "float16"
    assert runtime_versions["diffusion_model_source"]["revision"] == config.model_revision
    assert runtime_versions["sd35_operator_identity"] == {
        "component_class_names": {
            "pipeline": config.pipeline_class_name,
            "vae": config.vae_class_name,
            "transformer": config.transformer_class_name,
            "scheduler": config.scheduler_class_name,
        },
        "vae_scaling_factor": 1.5305,
        "vae_shift_factor": 0.0609,
        "latent_component_dtypes": {
            "vae": "float16",
            "transformer": "float16",
        },
    }
    pipeline.vae.config.scaling_factor = 1.0
    with pytest.raises(RuntimeError, match="scaling_factor"):
        sd3_pipeline_runtime._validate_loaded_pipeline(config, pipeline)


@pytest.mark.quick
def test_clip_loader_forwards_registered_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLIP 加载边界必须把精确 revision 传给 from_pretrained."""

    captured: dict[str, object] = {}

    class FakeParameter:
        """提供冻结参数所需接口."""

        def requires_grad_(self, enabled: bool) -> None:
            """记录梯度开关."""

            captured["requires_grad"] = enabled

    class FakeVisionModel:
        """记录 CLIP 加载参数而不下载权重."""

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> "FakeVisionModel":
            """保存模型标识和关键字参数."""

            captured["model_id"] = model_id
            captured.update(kwargs)
            return cls()

        def to(self, device_name: str) -> "FakeVisionModel":
            """记录目标设备并返回自身."""

            captured["device_name"] = device_name
            return self

        def eval(self) -> None:
            """记录推理模式."""

            captured["eval"] = True

        def parameters(self) -> tuple[FakeParameter, ...]:
            """返回一个可冻结的测试参数."""

            return (FakeParameter(),)

    fake_transformers = SimpleNamespace(CLIPVisionModelWithProjection=FakeVisionModel)
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    source = get_model_source("openai_clip_vit_base_patch32")

    semantic_model_loader.load_clip_vision_model(
        source.repository_id,
        source.revision,
        "cpu",
    )

    assert captured["model_id"] == source.repository_id
    assert captured["revision"] == source.revision
    assert captured["attn_implementation"] == "eager"
    assert captured["requires_grad"] is False
