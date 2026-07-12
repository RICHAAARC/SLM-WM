"""运行真实语义安全子空间嵌入和仅图像检测闭环。

该 runner 属于核心方法复现层, 在真实 SD3/SD3.5 latent 上计算分支风险、
完整特征 JVP/VJP Null Space、安全投影、真实 Q/K 注意力梯度和最终图像盲检。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

from experiments.protocol.method_runtime_config import load_formal_method_runtime_config
from experiments.runtime.diffusion.semantic_features import (
    DifferentiableSemanticFeatureRuntime,
    load_clip_vision_model,
)
from experiments.protocol.attacks import attack_config_digest, default_attack_configs
from experiments.runtime.diffusion.regeneration_attacks import (
    DiffusionAttackRuntime,
    default_diffusion_attack_specs,
)
from experiments.runtime.image_attacks import apply_standard_image_attack
from experiments.runtime.diffusion.sd3_pipeline_runtime import load_pipeline, tensor_norm
from experiments.runtime.image_metrics import compute_image_quality_metrics
from experiments.runtime.model_sources import (
    require_registered_model_reference,
)
from experiments.runtime.repository_environment import file_digest, resolve_code_version
from experiments.runtime.resume_checkpoint import (
    persist_completed_unit_from_manifest,
)
from experiments.runtime.scientific_unit_provenance import (
    build_scientific_unit_provenance,
    validate_scientific_unit_provenance,
)
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from main.core.digest import (
    TENSOR_CONTENT_DIGEST_VERSION,
    build_stable_digest,
    tensor_content_sha256,
)
from main.methods.carrier import (
    KEYED_PRG_VERSION,
    build_low_frequency_template,
    build_tail_robust_template,
    keyed_prg_protocol_record,
    project_canonical_template,
    require_supported_keyed_prg_version,
)
from main.methods.detection import ImageOnlyDetectionConfig, detect_image_only_watermark
from main.methods.geometry import (
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    ATTENTION_RELATION_COMPONENT_NAMES,
    DIRECT_QK_RELATION_SOURCE,
    DifferentiableAttentionRecorder,
    attention_geometry_component_scores,
    attention_geometry_score,
    attention_relation_component_protocol,
    attention_relation_stability_map,
    build_attention_relation_graph_identity,
    build_stable_attention_pair_weights,
    compute_attention_geometry_gradient,
    optimize_attention_geometry_update,
    qk_atomic_evaluation_records_digest,
    qk_atomic_evaluation_records_ready,
    select_stable_attention_tokens,
    validate_attention_relation_component_weights,
)
from main.methods.method_definition import (
    semantic_conditioned_latent_method_definition,
    semantic_conditioned_latent_method_definition_digest,
)
from main.methods.semantic import (
    BRANCH_NAMES,
    BranchRiskConfig,
    build_branch_risk_fields,
)
from main.methods.subspace import (
    ExactJacobianLinearization,
    build_exact_jacobian_linearization,
    exact_jvp,
    generate_keyed_candidate_directions,
    solve_jacobian_null_space,
)


_FORMAL_METHOD_CONFIG = load_formal_method_runtime_config(".")


@dataclass(frozen=True)
class SemanticWatermarkRuntimeConfig:
    """定义一次真实方法嵌入和仅图像检测运行。"""

    model_family: str = _FORMAL_METHOD_CONFIG.model_family
    model_id: str = _FORMAL_METHOD_CONFIG.model_id
    model_revision: str = _FORMAL_METHOD_CONFIG.model_revision
    vision_model_id: str = _FORMAL_METHOD_CONFIG.vision_model_id
    vision_model_revision: str = _FORMAL_METHOD_CONFIG.vision_model_revision
    device_name: str = "cuda"
    torch_dtype: str = "float16"
    vision_torch_dtype: str = "float32"
    hf_token_env: str = "HF_TOKEN"
    prompt: str = _FORMAL_METHOD_CONFIG.prompt
    prompt_id: str = "runtime_prompt"
    split: str = "dev"
    negative_prompt: str = _FORMAL_METHOD_CONFIG.negative_prompt
    key_material: str = "slm_wm_runtime_key"
    seed: int = _FORMAL_METHOD_CONFIG.seed
    width: int = _FORMAL_METHOD_CONFIG.width
    height: int = _FORMAL_METHOD_CONFIG.height
    inference_steps: int = _FORMAL_METHOD_CONFIG.inference_steps
    guidance_scale: float = _FORMAL_METHOD_CONFIG.guidance_scale
    injection_step_indices: tuple[int, ...] = _FORMAL_METHOD_CONFIG.injection_step_indices
    candidate_count: int = _FORMAL_METHOD_CONFIG.jacobian_candidate_count
    null_rank: int = _FORMAL_METHOD_CONFIG.null_space_rank
    lf_relative_strength: float = _FORMAL_METHOD_CONFIG.lf_relative_strength
    tail_relative_strength: float = _FORMAL_METHOD_CONFIG.tail_relative_strength
    attention_relative_strength: float = _FORMAL_METHOD_CONFIG.attention_relative_strength
    attention_stable_token_fraction: float = (
        _FORMAL_METHOD_CONFIG.attention_stable_token_fraction
    )
    attention_unstable_pair_weight: float = (
        _FORMAL_METHOD_CONFIG.attention_unstable_pair_weight
    )
    attention_relation_component_weights: tuple[float, ...] = (
        _FORMAL_METHOD_CONFIG.attention_relation_component_weights
    )
    minimum_final_image_attention_score_gain: float = (
        _FORMAL_METHOD_CONFIG.minimum_final_image_attention_score_gain
    )
    tail_fraction: float = _FORMAL_METHOD_CONFIG.tail_fraction
    keyed_prg_version: str = _FORMAL_METHOD_CONFIG.keyed_prg_version
    minimum_projection_energy_retention: float = _FORMAL_METHOD_CONFIG.minimum_projection_energy_retention
    maximum_relative_response_residual: float = _FORMAL_METHOD_CONFIG.maximum_relative_response_residual
    maximum_quantized_write_relative_jacobian_response: float = (
        _FORMAL_METHOD_CONFIG.maximum_quantized_write_relative_jacobian_response
    )
    null_space_cg_max_iterations: int = (
        _FORMAL_METHOD_CONFIG.null_space_cg_max_iterations
    )
    null_space_cg_relative_tolerance: float = (
        _FORMAL_METHOD_CONFIG.null_space_cg_relative_tolerance
    )
    minimum_semantic_preservation_cosine: float = (
        _FORMAL_METHOD_CONFIG.minimum_semantic_preservation_cosine
    )
    maximum_handcrafted_structure_feature_relative_drift: float = (
        _FORMAL_METHOD_CONFIG.maximum_handcrafted_structure_feature_relative_drift
    )
    max_attention_tokens: int = _FORMAL_METHOD_CONFIG.max_attention_tokens
    attention_module_names: tuple[str, ...] = (
        _FORMAL_METHOD_CONFIG.attention_module_names
    )
    attention_coordinate_convention: str = (
        _FORMAL_METHOD_CONFIG.attention_coordinate_convention
    )
    attention_grid_align_corners: bool = (
        _FORMAL_METHOD_CONFIG.attention_grid_align_corners
    )
    semantic_routing_enabled: bool = True
    branch_risk_mode: str = "branch_specific"
    null_space_enabled: bool = True
    lf_enabled: bool = True
    tail_robust_enabled: bool = True
    tail_truncation_enabled: bool = True
    attention_geometry_enabled: bool = True
    image_alignment_enabled: bool = True
    standard_attack_profiles: tuple[str, ...] = ("full_main",)
    diffusion_attacks_enabled: bool = _FORMAL_METHOD_CONFIG.diffusion_attacks_enabled
    content_threshold: float = 0.0
    geometry_score_threshold: float = 0.0
    registration_confidence_threshold: float = 0.0
    attention_sync_score_threshold: float = 0.0
    rescue_margin_low: float = -0.05
    output_dir: str = "outputs/semantic_watermark_runtime"

    def __post_init__(self) -> None:
        """集中校验重型运行配置。"""

        require_registered_model_reference(
            self.model_id,
            self.model_revision,
            required_usage_role="primary_diffusion_model",
        )
        require_registered_model_reference(
            self.vision_model_id,
            self.vision_model_revision,
            required_usage_role="semantic_condition_encoder",
        )
        if self.device_name != "cuda":
            raise ValueError("正式真实方法运行要求 CUDA 设备")
        if self.candidate_count < self.null_rank or self.null_rank <= 0:
            raise ValueError("candidate_count 必须不小于正的 null_rank")
        if any(
            index <= 0 or index >= self.inference_steps - 1
            for index in self.injection_step_indices
        ):
            raise ValueError(
                "post-step 注入时刻必须保留相邻的前后调度时刻"
            )
        if not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction 必须位于 (0, 1]")
        require_supported_keyed_prg_version(self.keyed_prg_version)
        if not 0.0 < self.attention_stable_token_fraction <= 1.0:
            raise ValueError(
                "attention_stable_token_fraction 必须位于 (0, 1]"
            )
        if not 0.0 <= self.attention_unstable_pair_weight < 1.0:
            raise ValueError(
                "attention_unstable_pair_weight 必须位于 [0, 1)"
            )
        validate_attention_relation_component_weights(
            self.attention_relation_component_weights
        )
        if (
            not math.isfinite(self.minimum_final_image_attention_score_gain)
            or self.minimum_final_image_attention_score_gain <= 0.0
        ):
            raise ValueError(
                "minimum_final_image_attention_score_gain 必须为正有限数"
            )
        if not 0.0 < self.minimum_projection_energy_retention <= 1.0:
            raise ValueError("minimum_projection_energy_retention 必须位于 (0, 1]")
        if not 0.0 < self.maximum_relative_response_residual <= 1.0:
            raise ValueError("maximum_relative_response_residual 必须位于 (0, 1]")
        if not 0.0 < self.maximum_quantized_write_relative_jacobian_response <= 1.0:
            raise ValueError(
                "maximum_quantized_write_relative_jacobian_response 必须位于 (0, 1]"
            )
        if self.null_space_cg_max_iterations <= 0:
            raise ValueError("null_space_cg_max_iterations 必须为正整数")
        if not 0.0 < self.null_space_cg_relative_tolerance < 1.0:
            raise ValueError("null_space_cg_relative_tolerance 必须位于 (0, 1)")
        if not 0.0 < self.minimum_semantic_preservation_cosine <= 1.0:
            raise ValueError(
                "minimum_semantic_preservation_cosine 必须位于 (0, 1]"
            )
        if not (
            0.0
            <= self.maximum_handcrafted_structure_feature_relative_drift
            <= 1.0
        ):
            raise ValueError(
                "maximum_handcrafted_structure_feature_relative_drift 必须位于 [0, 1]"
            )
        if self.branch_risk_mode not in {"branch_specific", "shared_global"}:
            raise ValueError(
                "branch_risk_mode 必须为 branch_specific 或 shared_global"
            )
        if not self.lf_enabled and not self.tail_robust_enabled:
            raise ValueError("正式内容检测至少需要启用一个内容载体分支")
        if len(self.attention_module_names) < 2:
            raise ValueError("真实注意力关系稳定度至少需要两个 Q/K 注意力层")
        if len(set(self.attention_module_names)) != len(
            self.attention_module_names
        ):
            raise ValueError("attention_module_names 不得包含重复层名")
        if self.attention_module_names != (
            _FORMAL_METHOD_CONFIG.attention_module_names
        ):
            raise ValueError("正式运行不得改变冻结的精确注意力层集合")
        if (
            self.attention_coordinate_convention
            != ATTENTION_COORDINATE_CONVENTION
            or self.attention_grid_align_corners
            is not ATTENTION_GRID_ALIGN_CORNERS
        ):
            raise ValueError("注意力 token 与图像坐标约定必须匹配核心算子")
        if self.max_attention_tokens < 4:
            raise ValueError("max_attention_tokens 至少为 4")
        if self.split not in {"dev", "calibration", "test"}:
            raise ValueError("split 必须为 dev、calibration 或 test")

    @property
    def carrier_model_reference(self) -> str:
        """返回同时绑定仓库和精确 revision 的公开载体标识."""

        return f"{self.model_id}@{self.model_revision}"


@dataclass(frozen=True)
class SemanticWatermarkRuntimeResult:
    """保存真实嵌入、图像输出和仅图像检测摘要。"""

    run_id: str
    run_decision: str
    clean_image_path: str
    watermarked_image_path: str
    update_record_path: str
    detection_record_path: str
    manifest_path: str
    update_count: int
    clean_detection_positive: bool
    watermarked_detection_positive: bool
    elapsed_seconds: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""

        return asdict(self)


@dataclass
class SemanticWatermarkRuntimeContext:
    """保存可跨 Prompt 复用的真实模型和特征运行时。"""

    pipeline: Any
    feature_runtime: DifferentiableSemanticFeatureRuntime
    attention_modules: tuple[tuple[str, Any], ...]
    unconditional_prompt: Any
    unconditional_pooled: Any
    runtime_versions: dict[str, Any]
    diffusion_attack_runtime: DiffusionAttackRuntime | None


def load_semantic_watermark_runtime_context(
    config: SemanticWatermarkRuntimeConfig,
) -> SemanticWatermarkRuntimeContext:
    """一次加载 SD3/SD3.5、CLIP 和注意力模块, 供数据集运行复用。"""

    pipeline, runtime_versions = load_pipeline(config)
    from diffusers.models.attention_processor import AttnProcessor

    pipeline.vae.set_attn_processor(AttnProcessor())
    runtime_versions["scientific_autograd_compatibility"] = {
        "clip_attention_implementation": "eager",
        "vae_attention_processor": "AttnProcessor",
        "reason": "exact_forward_ad_and_input_gradient_compatibility",
    }
    vision_model = load_clip_vision_model(
        config.vision_model_id,
        config.vision_model_revision,
        config.device_name,
        config.vision_torch_dtype,
    )
    runtime_versions["vision_model_source"] = require_registered_model_reference(
        config.vision_model_id,
        config.vision_model_revision,
        required_usage_role="semantic_condition_encoder",
    ).to_dict()
    feature_runtime = DifferentiableSemanticFeatureRuntime(pipeline.vae, vision_model)
    for parameter in pipeline.transformer.parameters():
        parameter.requires_grad_(False)
    attention_modules = _attention_modules(
        pipeline,
        config.attention_module_names,
    )
    runtime_versions["attention_operator_contract"] = {
        "attention_module_names": list(config.attention_module_names),
        "attention_coordinate_convention": (
            config.attention_coordinate_convention
        ),
        "attention_grid_align_corners": (
            config.attention_grid_align_corners
        ),
    }
    unconditional_prompt, unconditional_pooled = _unconditional_embeddings(
        pipeline,
        pipeline._execution_device,
    )
    diffusion_attack_runtime = (
        DiffusionAttackRuntime.from_text_to_image_pipeline(pipeline, config)
        if config.diffusion_attacks_enabled
        else None
    )
    return SemanticWatermarkRuntimeContext(
        pipeline=pipeline,
        feature_runtime=feature_runtime,
        attention_modules=attention_modules,
        unconditional_prompt=unconditional_prompt,
        unconditional_pooled=unconditional_pooled,
        runtime_versions=runtime_versions,
        diffusion_attack_runtime=diffusion_attack_runtime,
    )


def _stable_json(value: Any) -> str:
    """生成稳定 JSON 文本。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _is_sha256_hex(value: Any) -> bool:
    """判断字段是否为规范的小写 SHA-256 十六进制文本。"""

    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def semantic_watermark_runtime_config_payload(
    config: SemanticWatermarkRuntimeConfig,
) -> dict[str, Any]:
    """返回隐藏密钥原文但保留精确科学身份的运行配置."""

    payload = asdict(config)
    payload["key_material"] = build_stable_digest({"key_material": config.key_material})
    payload["injection_step_indices"] = list(config.injection_step_indices)
    payload["attention_module_names"] = list(config.attention_module_names)
    payload["attention_relation_component_weights"] = list(
        config.attention_relation_component_weights
    )
    payload["standard_attack_profiles"] = list(config.standard_attack_profiles)
    payload["method_definition"] = semantic_conditioned_latent_method_definition()
    payload["method_definition_digest"] = (
        semantic_conditioned_latent_method_definition_digest()
    )
    return payload


def semantic_watermark_runtime_config_digest(
    config: SemanticWatermarkRuntimeConfig,
) -> str:
    """计算单个 Prompt 或消融完成单元的配置摘要."""

    return build_stable_digest(semantic_watermark_runtime_config_payload(config))


def build_semantic_watermark_run_id(config: SemanticWatermarkRuntimeConfig) -> str:
    """根据完整运行配置生成稳定标识."""

    return f"semantic_watermark_{semantic_watermark_runtime_config_digest(config)[:16]}"


def validate_semantic_watermark_runtime_result_provenance(
    result_payload: Mapping[str, Any],
    *,
    expected_config: SemanticWatermarkRuntimeConfig | None = None,
) -> dict[str, Any]:
    """把持久化结果的配置、run id 与科学完成单元来源精确绑定."""

    payload = dict(result_payload)
    run_id = str(payload.get("run_id", ""))
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("语义水印完成结果缺少 metadata")
    unit_config = metadata.get("scientific_unit_config")
    if not isinstance(unit_config, Mapping):
        raise TypeError("语义水印完成结果缺少逐单元配置")
    resolved_unit_config = dict(unit_config)
    current_method_definition = semantic_conditioned_latent_method_definition()
    current_method_definition_digest = (
        semantic_conditioned_latent_method_definition_digest()
    )
    if (
        resolved_unit_config.get("method_definition")
        != current_method_definition
        or resolved_unit_config.get("method_definition_digest")
        != current_method_definition_digest
    ):
        raise ValueError("语义水印逐单元配置未绑定当前方法定义")
    config_digest = build_stable_digest(resolved_unit_config)
    if run_id != f"semantic_watermark_{config_digest[:16]}":
        raise ValueError("语义水印 run id 与逐单元配置摘要不一致")
    if (
        expected_config is not None
        and resolved_unit_config
        != semantic_watermark_runtime_config_payload(expected_config)
    ):
        raise ValueError("语义水印逐单元配置与当前请求不一致")
    provenance = metadata.get("scientific_unit_provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("语义水印完成结果缺少科学运行来源记录")
    return validate_scientific_unit_provenance(
        provenance,
        expected_unit_id=run_id,
        expected_config_digest=config_digest,
    )


def _carrier_only_counterfactual_artifact_binding_ready(
    result_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    root_path: Path,
    config: SemanticWatermarkRuntimeConfig,
) -> bool:
    """复验反事实原子、图像、保持记录、Q/K 记录与 manifest 绑定。"""

    if not config.attention_geometry_enabled:
        return True
    metadata = result_payload.get("metadata")
    manifest_metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping) or not isinstance(
        manifest_metadata,
        Mapping,
    ):
        return False
    observability = metadata.get("final_image_attention_observability")
    preservation = metadata.get("carrier_only_final_image_preservation")
    final_preservation = metadata.get("final_image_preservation")
    counterfactual = metadata.get("carrier_only_counterfactual")
    if not isinstance(observability, Mapping) or not isinstance(
        preservation,
        Mapping,
    ) or not isinstance(final_preservation, Mapping) or not isinstance(
        counterfactual,
        Mapping,
    ):
        return False
    identity_digest = str(
        counterfactual.get("carrier_only_counterfactual_identity_digest", "")
    )
    image_path = str(
        counterfactual.get("carrier_only_counterfactual_image_path", "")
    )
    image_digest = str(
        counterfactual.get("carrier_only_counterfactual_image_digest", "")
    )
    atom_path = str(
        counterfactual.get("carrier_only_counterfactual_atom_path", "")
    )
    atom_file_sha256 = str(
        counterfactual.get(
            "carrier_only_counterfactual_atom_file_sha256",
            "",
        )
    )
    atom_content_digest = str(
        counterfactual.get(
            "carrier_only_counterfactual_atom_content_digest",
            "",
        )
    )
    output_paths = tuple(str(path) for path in manifest.get("output_paths", ()))
    if not (
        _is_sha256_hex(identity_digest)
        and _is_sha256_hex(image_digest)
        and _is_sha256_hex(atom_file_sha256)
        and _is_sha256_hex(atom_content_digest)
        and image_path
        and atom_path
        and final_preservation.get("final_image_preservation_gate_ready") is True
        and preservation.get(
            "carrier_only_final_image_preservation_gate_ready"
        )
        is True
        and preservation.get(
            "carrier_only_to_full_final_image_preservation_gate_ready"
        )
        is True
        and preservation.get(
            "carrier_only_counterfactual_three_way_preservation_gate_ready"
        )
        is True
        and all(
            record.get("carrier_only_counterfactual_identity_digest")
            == identity_digest
            and record.get("carrier_only_counterfactual_image_path")
            == image_path
            and record.get("carrier_only_counterfactual_image_digest")
            == image_digest
            and record.get("carrier_only_counterfactual_atom_path")
            == atom_path
            and record.get("carrier_only_counterfactual_atom_file_sha256")
            == atom_file_sha256
            and record.get("carrier_only_counterfactual_atom_content_digest")
            == atom_content_digest
            for record in (observability, preservation)
        )
        and manifest_metadata.get("carrier_only_counterfactual_identity_digest")
        == identity_digest
        and manifest_metadata.get("carrier_only_counterfactual_image_digest")
        == image_digest
        and manifest_metadata.get("carrier_only_counterfactual_atom_path")
        == atom_path
        and manifest_metadata.get(
            "carrier_only_counterfactual_atom_file_sha256"
        )
        == atom_file_sha256
        and manifest_metadata.get(
            "carrier_only_counterfactual_atom_content_digest"
        )
        == atom_content_digest
        and image_path in output_paths
        and atom_path in output_paths
    ):
        return False

    def resolve_output_path(relative_path: str) -> Path | None:
        """把 manifest 相对路径限制在仓库根目录内。"""

        resolved = (root_path / relative_path).resolve()
        if resolved != root_path and root_path not in resolved.parents:
            return None
        return resolved

    resolved_image_path = resolve_output_path(image_path)
    resolved_atom_path = resolve_output_path(atom_path)
    full_record_path_text = str(result_payload.get("update_record_path", ""))
    resolved_full_record_path = resolve_output_path(full_record_path_text)
    if (
        resolved_image_path is None
        or resolved_atom_path is None
        or resolved_full_record_path is None
        or full_record_path_text not in output_paths
    ):
        return False
    if not (
        resolved_image_path.is_file()
        and resolved_atom_path.is_file()
        and resolved_full_record_path.is_file()
        and file_digest(resolved_image_path) == image_digest
        and file_digest(resolved_atom_path) == atom_file_sha256
    ):
        return False

    def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
        """读取更新原子并拒绝非对象行。"""

        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("更新原子必须是 JSON 对象")
            records.append(payload)
        return records

    try:
        full_records = read_jsonl_records(resolved_full_record_path)
        carrier_records = read_jsonl_records(resolved_atom_path)
        rebuilt_identity = _carrier_only_counterfactual_identity(
            config,
            replace(config, attention_geometry_enabled=False),
            full_records,
            carrier_records,
        )
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return bool(
        build_stable_digest(carrier_records) == atom_content_digest
        and rebuilt_identity["carrier_only_counterfactual_identity_digest"]
        == identity_digest
        and rebuilt_identity[
            "full_method_counterfactual_update_records_digest"
        ]
        == counterfactual.get(
            "full_method_counterfactual_update_records_digest"
        )
        and rebuilt_identity[
            "carrier_only_counterfactual_update_records_digest"
        ]
        == counterfactual.get(
            "carrier_only_counterfactual_update_records_digest"
        )
        and rebuilt_identity[
            "full_method_initial_latent_content_sha256"
        ]
        == counterfactual.get("full_method_initial_latent_content_sha256")
        and rebuilt_identity[
            "carrier_only_initial_latent_content_sha256"
        ]
        == counterfactual.get("carrier_only_initial_latent_content_sha256")
    )


def load_completed_semantic_watermark_runtime_result(
    config: SemanticWatermarkRuntimeConfig,
    root: str | Path = ".",
) -> SemanticWatermarkRuntimeResult | None:
    """读取同代码版本、同配置且文件完整的已完成运行。

    该函数用于 Colab 跨会话续跑。缓存只在运行决策为 pass、配置摘要一致、
    Git 代码版本一致且 manifest 中全部输出文件仍存在时复用, 避免把半写入
    目录或旧算法结果混入当前正式记录。
    """

    root_path = Path(root).resolve()
    run_id = build_semantic_watermark_run_id(config)
    run_dir = (root_path / config.output_dir / run_id).resolve()
    manifest_path = run_dir / "manifest.local.json"
    result_path = run_dir / "runtime_result.json"
    detection_path = run_dir / "image_only_detection_records.jsonl"
    if not manifest_path.is_file() or not result_path.is_file() or not detection_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_config_digest = semantic_watermark_runtime_config_digest(config)
    if manifest.get("config_digest") != expected_config_digest:
        return None
    if manifest.get("code_version") != resolve_code_version(root_path):
        return None
    if result_payload.get("run_decision") != "pass":
        return None
    try:
        validate_semantic_watermark_runtime_result_provenance(
            result_payload,
            expected_config=config,
        )
    except (TypeError, ValueError):
        return None
    output_paths = tuple(str(path) for path in manifest.get("output_paths", ()))
    if not output_paths or not all((root_path / path).is_file() for path in output_paths):
        return None
    if not _carrier_only_counterfactual_artifact_binding_ready(
        result_payload,
        manifest,
        root_path,
        config,
    ):
        return None
    try:
        return SemanticWatermarkRuntimeResult(**result_payload)
    except TypeError:
        return None


def _attention_modules(
    pipeline: Any,
    layer_names: tuple[str, ...],
) -> tuple[tuple[str, Any], ...]:
    """按配置中的精确层名解析真实 Q/K 注意力模块."""

    available = dict(pipeline.transformer.named_modules())
    resolved = []
    for layer_name in layer_names:
        module = available.get(layer_name)
        if module is None:
            raise RuntimeError(
                f"冻结注意力层不存在: {layer_name}"
            )
        if not all(
            hasattr(module, attribute)
            for attribute in ("to_q", "to_k", "heads")
        ):
            raise RuntimeError(
                f"冻结注意力层不满足公开 Q/K 协议: {layer_name}"
            )
        resolved.append((layer_name, module))
    return tuple(resolved)


def _unconditional_embeddings(pipeline: Any, device: Any) -> tuple[Any, Any]:
    """构造嵌入端和检测端都可复现的空文本条件。"""

    prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
        prompt="",
        prompt_2="",
        prompt_3="",
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    return prompt_embeds, pooled_prompt_embeds


def _transformer_forward_function(
    pipeline: Any,
    timestep: Any,
    prompt_embeds: Any,
    pooled_prompt_embeds: Any,
) -> Any:
    """构造以 latent 为唯一变量的真实 Transformer 前向函数。"""

    transformer_dtype = next(pipeline.transformer.parameters()).dtype

    def forward(latent: Any) -> Any:
        timestep_batch = timestep.expand(latent.shape[0])
        return pipeline.transformer(
            hidden_states=latent.to(dtype=transformer_dtype),
            timestep=timestep_batch,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return forward


def _branch_budget(signal_map: Any, branch_field: Any) -> tuple[float, ...]:
    """把 batch 风险图收敛为带硬资格边界的空间预算。"""

    spatial_count = int(signal_map.shape[-2] * signal_map.shape[-1])
    values = branch_field.budget_values
    if len(values) != spatial_count:
        raise RuntimeError("分支风险预算与 latent 空间网格不一致")
    eligible = set(int(index) for index in branch_field.eligible_indices)
    return tuple(value if index in eligible else 0.0 for index, value in enumerate(values))


def _active_carrier_branch_names(
    config: SemanticWatermarkRuntimeConfig,
) -> tuple[str, ...]:
    """返回当前机制配置中实际参与嵌入的载体分支。"""

    enabled = {
        "lf_content": config.lf_enabled,
        "tail_robust": config.tail_robust_enabled,
        "attention_geometry": config.attention_geometry_enabled,
    }
    return tuple(branch_name for branch_name, is_enabled in enabled.items() if is_enabled)


def _required_branch_risk_eligibility(
    config: SemanticWatermarkRuntimeConfig,
) -> tuple[str, ...]:
    """返回需要执行风险资格门禁的活动分支。

    完整方法只对实际参与嵌入的分支执行 fail-closed 门禁. 移除风险路由的正式
    消融返回空集合, 避免已移除机制继续筛掉高风险样本.
    """

    if not config.semantic_routing_enabled:
        return ()
    return _active_carrier_branch_names(config)


def _branch_risk_configs(
    config: SemanticWatermarkRuntimeConfig,
) -> dict[str, BranchRiskConfig] | None:
    """返回完整方法或共享全局风险正式消融使用的风险配置。

    ``None`` 让核心方法构造器使用三个分支各自的正式风险定义。
    ``shared_global`` 则把同一个预注册全局风险复制到三个载体分支, 用于直接
    检验分支特定风险语义是否优于单一共享路由。该对照仍执行真实风险门控,
    不等价于完全移除路由。
    """

    if config.branch_risk_mode == "branch_specific":
        return None
    shared_config = BranchRiskConfig(
        local_contrast_risk_weight=0.25,
        semantic_weight=0.25,
        texture_weight=0.20,
        adjacent_step_instability_weight=0.20,
        attention_instability_weight=0.10,
        texture_preference="avoid",
    )
    return {
        branch_name: shared_config
        for branch_name in (
            "lf_content",
            "tail_robust",
            "attention_geometry",
        )
    }


def _branch_risk_record(branch_field: Any) -> dict[str, Any]:
    """生成带精确风险、预算与资格 mask 内容摘要的分支记录."""

    return {
        "branch_name": branch_field.branch_name,
        "risk_field_digest": branch_field.risk_field_digest,
        "risk_values_content_sha256": (
            branch_field.risk_values_content_sha256
        ),
        "budget_values_content_sha256": (
            branch_field.budget_values_content_sha256
        ),
        "eligible_mask_content_sha256": (
            branch_field.eligible_mask_content_sha256
        ),
        "risk_value_mean": sum(branch_field.risk_values) / len(branch_field.risk_values),
        "budget_value_mean": sum(branch_field.budget_values) / len(branch_field.budget_values),
        "eligible_position_count": len(branch_field.eligible_indices),
        "risk_field_position_count": len(branch_field.risk_values),
        "metadata": branch_field.metadata,
    }


def _solve_branch_subspace(
    latent: Any,
    feature_runtime: DifferentiableSemanticFeatureRuntime,
    key_material: str,
    branch_name: str,
    axis_budget: tuple[float, ...],
    candidate_count: int,
    null_rank: int,
    joint_feature_linearization: ExactJacobianLinearization,
    preferred_directions: tuple[Any, ...] = (),
    maximum_relative_response_residual: float = 1e-4,
    minimum_projection_energy_retention: float = 0.01,
    cg_maximum_iterations: int = 64,
    cg_relative_tolerance: float = 1e-6,
    prg_version: str = KEYED_PRG_VERSION,
) -> Any:
    """为一个载体分支运行完整 Jacobian 风险支持约束投影。"""

    candidates = generate_keyed_candidate_directions(
        latent,
        key_material,
        branch_name,
        candidate_count,
        axis_budget=None,
        preferred_directions=preferred_directions,
        prg_version=prg_version,
    )
    result = solve_jacobian_null_space(
        latent=latent.float(),
        candidate_matrix=candidates,
        risk_budget=axis_budget,
        null_rank=null_rank,
        joint_feature_linearization=joint_feature_linearization,
        branch_name=branch_name,
        maximum_relative_response_residual=maximum_relative_response_residual,
        minimum_projection_energy_retention=minimum_projection_energy_retention,
        cg_maximum_iterations=cg_maximum_iterations,
        cg_relative_tolerance=cg_relative_tolerance,
    )
    result.metadata["preferred_direction_count"] = len(preferred_directions)
    result.metadata["preferred_direction_role"] = "carrier_or_attention_gradient"
    result.metadata.update(keyed_prg_protocol_record(prg_version))
    result.metadata.update(feature_runtime.feature_schema_record())
    if result.relative_response_residual > maximum_relative_response_residual:
        raise RuntimeError("完整 Jacobian Null Space 的相对响应残差超过正式门禁")
    return result


def _feature_preservation_values(
    semantic_before: Any,
    structure_before: Any,
    semantic_after: Any,
    structure_after: Any,
    config: SemanticWatermarkRuntimeConfig,
) -> tuple[float, float, bool]:
    """统一计算完整 CLIP 与手工结构统计的保持指标和门禁."""

    import torch
    import torch.nn.functional as functional

    semantic_before_flat = semantic_before.float().reshape(-1)
    semantic_after_flat = semantic_after.float().reshape(-1)
    structure_before_flat = structure_before.float().reshape(-1)
    structure_after_flat = structure_after.float().reshape(-1)
    if semantic_before_flat.shape != semantic_after_flat.shape:
        raise RuntimeError("写回前后完整 CLIP 特征宽度不一致")
    if structure_before_flat.shape != structure_after_flat.shape:
        raise RuntimeError("写回前后手工结构统计特征宽度不一致")
    semantic_cosine = float(
        functional.cosine_similarity(
            semantic_before_flat,
            semantic_after_flat,
            dim=0,
            eps=1e-12,
        ).item()
    )
    structure_relative_drift = float(
        torch.linalg.norm(structure_after_flat - structure_before_flat).item()
        / max(float(torch.linalg.norm(structure_before_flat).item()), 1e-12)
    )
    ready = bool(
        math.isfinite(semantic_cosine)
        and math.isfinite(structure_relative_drift)
        and semantic_cosine >= config.minimum_semantic_preservation_cosine
        and structure_relative_drift <= config.maximum_handcrafted_structure_feature_relative_drift
    )
    return semantic_cosine, structure_relative_drift, ready


def _quantized_write_jacobian_response_record(
    feature_function: Any | None,
    latent: Any,
    injected: Any,
    maximum_relative_response: float,
) -> dict[str, Any]:
    """复验实际量化写回 Tensor 的完整特征 Jacobian 响应。

    Null Space 基底在 float32 中求解, 但扩散 latent 通常使用 float16。该函数
    先按真实 latent dtype 完成加法, 再以 ``written_latent - latent`` 恢复实际
    写入增量。相对响应以当前完整特征向量二范数归一化, 因而直接表示一阶
    特征变化相对于当前语义与视觉状态的比例。该门禁验证量化后的写回对象,
    不能由量化前的分支方向或有限更新保持记录替代。
    """

    import torch

    if not 0.0 < maximum_relative_response <= 1.0:
        raise ValueError("实际写回 Jacobian 相对响应阈值必须位于 (0, 1]")
    if tuple(latent.shape) != tuple(injected.shape):
        raise ValueError("实际写回前后的 latent 形状必须一致")
    quantized_latent = injected.detach().to(dtype=latent.dtype)
    quantized_update = quantized_latent - latent.detach()
    update_norm = float(torch.linalg.norm(quantized_update.float()).item())
    base_record = {
        "quantized_write_update_content_sha256": tensor_content_sha256(
            quantized_update
        ),
        "quantized_write_update_dtype": str(quantized_update.dtype),
        "quantized_write_update_shape": [
            int(value) for value in quantized_update.shape
        ],
        "quantized_write_update_norm": update_norm,
        "maximum_quantized_write_relative_jacobian_response": (
            maximum_relative_response
        ),
    }
    if feature_function is None:
        return {
            **base_record,
            "quantized_write_jacobian_gate_applicable": False,
            "quantized_write_jacobian_response_norm": None,
            "quantized_write_reference_feature_norm": None,
            "quantized_write_relative_jacobian_response": None,
            "quantized_write_jacobian_gate_ready": False,
            "quantized_write_jacobian_status": (
                "not_applicable_jacobian_null_space_disabled"
            ),
        }
    primal, response = exact_jvp(
        feature_function,
        latent.detach().float(),
        quantized_update.float(),
    )
    response = response.detach().float()
    response_norm = float(torch.linalg.norm(response).item())
    reference_feature_norm = float(
        torch.linalg.norm(primal.detach().float()).item()
    )
    relative_response = response_norm / max(reference_feature_norm, 1e-12)
    ready = bool(
        math.isfinite(update_norm)
        and update_norm > 0.0
        and math.isfinite(response_norm)
        and math.isfinite(reference_feature_norm)
        and math.isfinite(relative_response)
        and relative_response <= maximum_relative_response
    )
    return {
        **base_record,
        "quantized_write_jacobian_gate_applicable": True,
        "quantized_write_jacobian_response_norm": response_norm,
        "quantized_write_reference_feature_norm": reference_feature_norm,
        "quantized_write_relative_jacobian_response": relative_response,
        "quantized_write_jacobian_gate_ready": ready,
        "quantized_write_jacobian_status": (
            "measured_from_actual_quantized_latent_delta"
        ),
    }


def _combined_update_preservation_record(
    feature_runtime: DifferentiableSemanticFeatureRuntime,
    latent: Any,
    injected: Any,
    config: SemanticWatermarkRuntimeConfig,
) -> dict[str, Any]:
    """验证一次实际写回 latent 的完整特征有限更新保持性。"""

    import torch

    with torch.no_grad():
        semantic_before, structure_before = feature_runtime.joint_features(
            latent.detach().float()
        )
        semantic_after, structure_after = feature_runtime.joint_features(
            injected.detach().float()
        )
    semantic_cosine, structure_relative_drift, ready = _feature_preservation_values(
        semantic_before,
        structure_before,
        semantic_after,
        structure_after,
        config,
    )
    return {
        "full_semantic_cosine_similarity": semantic_cosine,
        "full_handcrafted_structure_feature_relative_drift": structure_relative_drift,
        "minimum_semantic_preservation_cosine": (
            config.minimum_semantic_preservation_cosine
        ),
        "maximum_handcrafted_structure_feature_relative_drift": (
            config.maximum_handcrafted_structure_feature_relative_drift
        ),
        "semantic_preservation_gate_ready": ready,
        "preservation_validation_scope": (
            "actual_combined_latent_full_clip_and_handcrafted_structure_features"
        ),
    }


def _final_image_preservation_record(
    pipeline: Any,
    feature_runtime: DifferentiableSemanticFeatureRuntime,
    clean_image: Any,
    watermarked_image: Any,
    config: SemanticWatermarkRuntimeConfig,
) -> dict[str, Any]:
    """在最终成图上验证累计 CLIP 与手工结构统计保持性."""

    clean_features, watermarked_features = _final_image_joint_features(
        pipeline,
        feature_runtime,
        clean_image,
        watermarked_image,
    )
    semantic_cosine, structure_relative_drift, ready = _feature_preservation_values(
        clean_features[0],
        clean_features[1],
        watermarked_features[0],
        watermarked_features[1],
        config,
    )
    return {
        "final_image_semantic_cosine_similarity": semantic_cosine,
        "final_image_handcrafted_structure_feature_relative_drift": structure_relative_drift,
        "minimum_semantic_preservation_cosine": (
            config.minimum_semantic_preservation_cosine
        ),
        "maximum_handcrafted_structure_feature_relative_drift": (
            config.maximum_handcrafted_structure_feature_relative_drift
        ),
        "final_image_preservation_gate_ready": ready,
        "preservation_validation_scope": (
            "paired_final_images_full_clip_and_handcrafted_structure_features"
        ),
    }


def _final_image_joint_features(
    pipeline: Any,
    feature_runtime: DifferentiableSemanticFeatureRuntime,
    *images: Any,
) -> tuple[tuple[Any, Any], ...]:
    """一次性提取最终成图的完整 CLIP 与手工结构统计."""

    import torch

    device = pipeline._execution_device

    def image_tensor(image: Any) -> Any:
        """把最终 PIL 成图转换为 [0, 1] 模型输入 tensor。"""

        pixels = pipeline.image_processor.preprocess(image).to(
            device=device,
            dtype=torch.float32,
        )
        return (pixels / 2.0 + 0.5).clamp(0.0, 1.0)

    with torch.no_grad():
        return tuple(
            feature_runtime.joint_image_features(image_tensor(image))
            for image in images
        )


def _three_way_final_image_preservation_records(
    pipeline: Any,
    feature_runtime: DifferentiableSemanticFeatureRuntime,
    clean_image: Any,
    carrier_only_image: Any,
    watermarked_image: Any,
    config: SemanticWatermarkRuntimeConfig,
    carrier_only_counterfactual: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """验证 clean、carrier-only 与完整方法最终成图的三边保持性。"""

    identity_digest = str(
        carrier_only_counterfactual.get(
            "carrier_only_counterfactual_identity_digest",
            "",
        )
    )
    if (
        carrier_only_counterfactual.get("carrier_only_counterfactual_ready")
        is not True
        or len(identity_digest) != 64
    ):
        raise RuntimeError("carrier-only 最终保持门禁缺少反事实身份")
    clean_features, carrier_features, watermarked_features = (
        _final_image_joint_features(
            pipeline,
            feature_runtime,
            clean_image,
            carrier_only_image,
            watermarked_image,
        )
    )
    clean_full_values = _feature_preservation_values(
        clean_features[0],
        clean_features[1],
        watermarked_features[0],
        watermarked_features[1],
        config,
    )
    clean_carrier_values = _feature_preservation_values(
        clean_features[0],
        clean_features[1],
        carrier_features[0],
        carrier_features[1],
        config,
    )
    carrier_full_values = _feature_preservation_values(
        carrier_features[0],
        carrier_features[1],
        watermarked_features[0],
        watermarked_features[1],
        config,
    )
    final_record = {
        "final_image_semantic_cosine_similarity": clean_full_values[0],
        "final_image_handcrafted_structure_feature_relative_drift": clean_full_values[1],
        "minimum_semantic_preservation_cosine": (
            config.minimum_semantic_preservation_cosine
        ),
        "maximum_handcrafted_structure_feature_relative_drift": (
            config.maximum_handcrafted_structure_feature_relative_drift
        ),
        "final_image_preservation_gate_ready": clean_full_values[2],
        "preservation_validation_scope": (
            "clean_to_full_final_images_full_clip_and_handcrafted_structure_features"
        ),
    }
    counterfactual_record = {
        "carrier_only_final_image_preservation_applicable": True,
        "carrier_only_final_image_semantic_cosine_similarity": (
            clean_carrier_values[0]
        ),
        "carrier_only_final_image_handcrafted_structure_feature_relative_drift": (
            clean_carrier_values[1]
        ),
        "carrier_only_final_image_preservation_gate_ready": (
            clean_carrier_values[2]
        ),
        "carrier_only_to_full_final_image_semantic_cosine_similarity": (
            carrier_full_values[0]
        ),
        "carrier_only_to_full_final_image_handcrafted_structure_feature_relative_drift": (
            carrier_full_values[1]
        ),
        "carrier_only_to_full_final_image_preservation_gate_ready": (
            carrier_full_values[2]
        ),
        "carrier_only_counterfactual_three_way_preservation_gate_ready": (
            clean_full_values[2]
            and clean_carrier_values[2]
            and carrier_full_values[2]
        ),
        "minimum_semantic_preservation_cosine": (
            config.minimum_semantic_preservation_cosine
        ),
        "maximum_handcrafted_structure_feature_relative_drift": (
            config.maximum_handcrafted_structure_feature_relative_drift
        ),
        "carrier_only_counterfactual_identity_digest": identity_digest,
        "carrier_only_final_image_preservation_status": (
            "measured_from_clean_carrier_only_and_full_final_images"
        ),
        "preservation_validation_scope": (
            "three_pair_final_images_full_clip_and_handcrafted_structure_features"
        ),
    }
    return final_record, counterfactual_record


def _final_image_attention_attribution_gate_ready(
    *,
    blind_attribution_gain: float,
    frozen_pair_attribution_gain: float,
    minimum_gain: float,
    measured_values: tuple[float, ...],
    relation_identity_ready: bool,
) -> bool:
    """要求盲选择与冻结 carrier pair 两条归因证据同时通过。"""

    return bool(
        relation_identity_ready
        and all(math.isfinite(value) for value in measured_values)
        and blind_attribution_gain > minimum_gain
        and frozen_pair_attribution_gain > minimum_gain
    )


def _final_image_attention_observability_record(
    image_attention_extractor: Any | None,
    clean_image: Any,
    carrier_only_image: Any | None,
    watermarked_image: Any,
    config: SemanticWatermarkRuntimeConfig,
    *,
    carrier_only_counterfactual: Mapping[str, Any] | None = None,
    require_gpu_execution: bool = True,
) -> dict[str, Any]:
    """以 carrier-only 反事实验证最终成图中的 attention 因果增益。

    clean 只保留为总体水印对照。正式 attention 归因比较同 seed、同 scheduler、
    同 LF/tail 配置与算子的 carrier-only 图像与完整方法图像, 估计含下游交互的
    总机制效应, 不假设两侧已经实现相同 carrier。盲分数允许两张图分别执行自身
    稳定 token 选择; 配对分数则冻结 carrier-only 的 pair 权重。两种归因增益都
    必须严格超过正下界, 且不得使用中间 latent 分数替代最终成图 Q/K。
    """

    source = "image_reencoded_public_noise_real_qk"
    if not config.attention_geometry_enabled:
        return {
            "final_image_attention_observability_applicable": False,
            "final_image_attention_observability_gate_ready": False,
            "final_image_attention_observability_source": source,
            "final_image_attention_observability_requires_gpu": True,
            "final_image_attention_observability_gpu_execution_verified": False,
            "minimum_final_image_attention_score_gain": (
                config.minimum_final_image_attention_score_gain
            ),
            "final_clean_blind_attention_score": None,
            "final_carrier_only_blind_attention_score": None,
            "final_watermarked_blind_attention_score": None,
            "final_image_blind_attention_score_gain": None,
            "final_image_attention_blind_attribution_gain": None,
            "final_clean_paired_attention_score": None,
            "final_carrier_only_paired_attention_score": None,
            "final_watermarked_carrier_paired_attention_score": None,
            "final_watermarked_paired_attention_score": None,
            "final_image_paired_attention_score_gain": None,
            "final_image_attention_carrier_paired_attribution_gain": None,
            "final_clean_pair_weight_identity_digest": "",
            "final_carrier_only_pair_weight_identity_digest": "",
            "final_watermarked_pair_weight_identity_digest": "",
            "final_paired_pair_weight_identity_digest": "",
            "final_image_attention_record_schema_digest": "",
            "attention_relation_component_names": [],
            "attention_relation_active_component_names": [],
            "attention_relation_component_weights": [],
            "attention_relation_component_protocol_digest": "",
            "attention_relation_source": "",
            "attention_relation_direct_qk_source_ready": False,
            "attention_relation_component_identity_digest": "",
            "attention_relation_keyed_projection_digest": "",
            "attention_relation_qk_operator_metadata_records": [],
            "attention_relation_qk_operator_metadata_digest": "",
            "attention_relation_qk_operator_metadata_ready": False,
            "final_image_qk_atomic_content_records": [],
            "final_image_qk_atomic_content_digest": "",
            "final_image_qk_atomic_content_ready": False,
            "attention_module_names": list(config.attention_module_names),
            "attention_coordinate_convention": (
                config.attention_coordinate_convention
            ),
            "attention_grid_align_corners": (
                config.attention_grid_align_corners
            ),
            "final_carrier_only_paired_attention_component_scores": {},
            "final_watermarked_carrier_paired_attention_component_scores": {},
            "final_image_attention_carrier_paired_component_gains": {},
            "carrier_only_counterfactual_ready": False,
            "observability_status": "not_applicable_attention_geometry_disabled",
        }
    if image_attention_extractor is None:
        raise RuntimeError("最终成图注意力可观测性门禁缺少真实 Q/K 提取器")
    if carrier_only_image is None or not carrier_only_counterfactual:
        raise RuntimeError("最终成图注意力归因缺少 carrier-only 反事实")
    if carrier_only_counterfactual.get("carrier_only_counterfactual_ready") is not True:
        raise RuntimeError("carrier-only 反事实身份没有通过同种子同调度门禁")

    clean_records = tuple(image_attention_extractor(clean_image))
    carrier_only_records = tuple(image_attention_extractor(carrier_only_image))
    watermarked_records = tuple(image_attention_extractor(watermarked_image))
    if any(
        len(records) < 2
        for records in (clean_records, carrier_only_records, watermarked_records)
    ):
        raise RuntimeError("最终成图注意力可观测性要求至少两个真实 Q/K 层")

    def record_schema(records: tuple[Any, ...]) -> tuple[Any, ...]:
        """返回 Q/K 层名称与二维 token 网格的共同身份。"""

        return tuple(
            (layer_name, tuple(token_indices))
            for layer_name, _, token_indices in records
        )

    clean_record_schema = record_schema(clean_records)
    carrier_only_record_schema = record_schema(carrier_only_records)
    watermarked_record_schema = record_schema(watermarked_records)
    if not (
        clean_record_schema
        == carrier_only_record_schema
        == watermarked_record_schema
    ):
        raise RuntimeError("最终三图 Q/K 层身份或二维网格不一致")
    if tuple(name for name, _ in clean_record_schema) != (
        config.attention_module_names
    ):
        raise RuntimeError("最终成图 Q/K 记录没有使用配置冻结的精确层名")
    relation_identities = tuple(
        build_attention_relation_graph_identity(
            records,
            config.key_material,
            config.attention_relation_component_weights,
        )
        for records in (
            clean_records,
            carrier_only_records,
            watermarked_records,
        )
    )
    relation_identity = relation_identities[0]
    relation_identity_ready = all(
        identity.relation_source == DIRECT_QK_RELATION_SOURCE
        and identity.qk_operator_metadata_ready
        and identity.qk_atomic_content_ready
        and identity.component_names == relation_identity.component_names
        and identity.active_component_names
        == relation_identity.active_component_names
        and identity.component_weights == relation_identity.component_weights
        and identity.component_protocol_digest
        == relation_identity.component_protocol_digest
        and identity.component_identity_digest
        == relation_identity.component_identity_digest
        and identity.keyed_projection_digest
        == relation_identity.keyed_projection_digest
        and identity.qk_operator_metadata_digest
        == relation_identity.qk_operator_metadata_digest
        for identity in relation_identities
    )
    if not relation_identity_ready:
        raise RuntimeError("最终三图没有共享直接 Q/K 四分量关系图身份")
    final_image_qk_atomic_content_records = tuple(
        {
            "qk_evaluation_role": evaluation_role,
            "qk_atomic_content_records": list(
                identity.qk_atomic_content_records
            ),
            "qk_atomic_content_digest": identity.qk_atomic_content_digest,
            "qk_atomic_content_ready": identity.qk_atomic_content_ready,
        }
        for evaluation_role, identity in zip(
            (
                "final_clean_image",
                "final_carrier_only_image",
                "final_watermarked_image",
            ),
            relation_identities,
        )
    )
    all_records = clean_records + carrier_only_records + watermarked_records
    gpu_verified = all(
        getattr(attention, "device", None) is not None
        and attention.device.type == "cuda"
        for _, attention, _ in all_records
    )
    if require_gpu_execution and not gpu_verified:
        raise RuntimeError("最终成图真实 Q/K 可观测性必须在 CUDA 上执行")

    clean_selection = select_stable_attention_tokens(
        clean_records,
        stable_token_fraction=config.attention_stable_token_fraction,
    )
    carrier_only_selection = select_stable_attention_tokens(
        carrier_only_records,
        stable_token_fraction=config.attention_stable_token_fraction,
    )
    watermarked_selection = select_stable_attention_tokens(
        watermarked_records,
        stable_token_fraction=config.attention_stable_token_fraction,
    )
    clean_pair_weights = build_stable_attention_pair_weights(
        clean_records,
        clean_selection,
        unstable_pair_weight=config.attention_unstable_pair_weight,
    )
    carrier_only_pair_weights = build_stable_attention_pair_weights(
        carrier_only_records,
        carrier_only_selection,
        unstable_pair_weight=config.attention_unstable_pair_weight,
    )
    watermarked_pair_weights = build_stable_attention_pair_weights(
        watermarked_records,
        watermarked_selection,
        unstable_pair_weight=config.attention_unstable_pair_weight,
    )

    def score(records: tuple[Any, ...], pair_weights: Any) -> float:
        """以显式冻结 pair 权重计算最终成图真实 Q/K 分数。"""

        value = attention_geometry_score(
            records,
            config.key_material,
            stable_pair_weights=pair_weights,
            component_weights=config.attention_relation_component_weights,
        )
        return float(value.detach().item())

    clean_blind_score = score(clean_records, clean_pair_weights)
    carrier_only_blind_score = score(
        carrier_only_records,
        carrier_only_pair_weights,
    )
    watermarked_blind_score = score(
        watermarked_records,
        watermarked_pair_weights,
    )
    clean_paired_score = score(clean_records, clean_pair_weights)
    watermarked_paired_score = score(watermarked_records, clean_pair_weights)
    carrier_only_paired_score = score(
        carrier_only_records,
        carrier_only_pair_weights,
    )
    watermarked_carrier_paired_score = score(
        watermarked_records,
        carrier_only_pair_weights,
    )
    carrier_only_paired_components = attention_geometry_component_scores(
        carrier_only_records,
        config.key_material,
        carrier_only_pair_weights,
        config.attention_relation_component_weights,
    )
    watermarked_carrier_paired_components = attention_geometry_component_scores(
        watermarked_records,
        config.key_material,
        carrier_only_pair_weights,
        config.attention_relation_component_weights,
    )
    carrier_paired_component_gains = (
        watermarked_carrier_paired_components - carrier_only_paired_components
    )
    clean_control_blind_gain = watermarked_blind_score - clean_blind_score
    clean_control_paired_gain = watermarked_paired_score - clean_paired_score
    blind_attribution_gain = watermarked_blind_score - carrier_only_blind_score
    carrier_paired_attribution_gain = (
        watermarked_carrier_paired_score - carrier_only_paired_score
    )
    values = (
        clean_blind_score,
        carrier_only_blind_score,
        watermarked_blind_score,
        clean_paired_score,
        carrier_only_paired_score,
        watermarked_carrier_paired_score,
        watermarked_paired_score,
        clean_control_blind_gain,
        clean_control_paired_gain,
        blind_attribution_gain,
        carrier_paired_attribution_gain,
    )
    ready = _final_image_attention_attribution_gate_ready(
        blind_attribution_gain=blind_attribution_gain,
        frozen_pair_attribution_gain=carrier_paired_attribution_gain,
        minimum_gain=config.minimum_final_image_attention_score_gain,
        measured_values=values,
        relation_identity_ready=relation_identity_ready,
    )
    return {
        **dict(carrier_only_counterfactual),
        "final_image_attention_observability_applicable": True,
        "final_image_attention_observability_gate_ready": ready,
        "final_image_attention_observability_source": source,
        "final_image_attention_observability_requires_gpu": True,
        "final_image_attention_observability_gpu_execution_verified": gpu_verified,
        "minimum_final_image_attention_score_gain": (
            config.minimum_final_image_attention_score_gain
        ),
        "final_clean_blind_attention_score": clean_blind_score,
        "final_carrier_only_blind_attention_score": carrier_only_blind_score,
        "final_watermarked_blind_attention_score": watermarked_blind_score,
        "final_image_blind_attention_score_gain": clean_control_blind_gain,
        "final_image_attention_blind_attribution_gain": blind_attribution_gain,
        "final_clean_paired_attention_score": clean_paired_score,
        "final_carrier_only_paired_attention_score": carrier_only_paired_score,
        "final_watermarked_carrier_paired_attention_score": (
            watermarked_carrier_paired_score
        ),
        "final_watermarked_paired_attention_score": watermarked_paired_score,
        "final_image_paired_attention_score_gain": clean_control_paired_gain,
        "final_image_attention_carrier_paired_attribution_gain": (
            carrier_paired_attribution_gain
        ),
        "final_clean_pair_weight_identity_digest": (
            clean_pair_weights.pair_weight_identity_digest
        ),
        "final_carrier_only_pair_weight_identity_digest": (
            carrier_only_pair_weights.pair_weight_identity_digest
        ),
        "final_watermarked_pair_weight_identity_digest": (
            watermarked_pair_weights.pair_weight_identity_digest
        ),
        "final_paired_pair_weight_identity_digest": (
            clean_pair_weights.pair_weight_identity_digest
        ),
        "final_image_attention_record_schema_digest": build_stable_digest(
            {"attention_record_schema": clean_record_schema}
        ),
        "attention_relation_component_names": list(
            relation_identity.component_names
        ),
        "attention_relation_active_component_names": list(
            relation_identity.active_component_names
        ),
        "attention_relation_component_weights": list(
            relation_identity.component_weights
        ),
        "attention_relation_component_protocol_digest": (
            relation_identity.component_protocol_digest
        ),
        "attention_relation_source": relation_identity.relation_source,
        "attention_relation_direct_qk_source_ready": relation_identity_ready,
        "attention_relation_probability_scope": (
            "sampled_image_token_qk_relation_probability"
        ),
        "attention_relation_component_identity_digest": (
            relation_identity.component_identity_digest
        ),
        "attention_relation_keyed_projection_digest": (
            relation_identity.keyed_projection_digest
        ),
        "attention_relation_qk_operator_metadata_records": list(
            relation_identity.qk_operator_metadata_records
        ),
        "attention_relation_qk_operator_metadata_digest": (
            relation_identity.qk_operator_metadata_digest
        ),
        "attention_relation_qk_operator_metadata_ready": (
            relation_identity.qk_operator_metadata_ready
        ),
        "final_image_qk_atomic_content_records": list(
            final_image_qk_atomic_content_records
        ),
        "final_image_qk_atomic_content_digest": (
            qk_atomic_evaluation_records_digest(
                final_image_qk_atomic_content_records,
                "final_image_qk_atomic_content_records",
            )
        ),
        "final_image_qk_atomic_content_ready": all(
            bool(record["qk_atomic_content_ready"])
            for record in final_image_qk_atomic_content_records
        ),
        "attention_module_names": list(config.attention_module_names),
        "attention_coordinate_convention": (
            config.attention_coordinate_convention
        ),
        "attention_grid_align_corners": (
            config.attention_grid_align_corners
        ),
        "final_carrier_only_paired_attention_component_scores": {
            name: float(value)
            for name, value in zip(
                ATTENTION_RELATION_COMPONENT_NAMES,
                carrier_only_paired_components.detach().cpu().tolist(),
            )
        },
        "final_watermarked_carrier_paired_attention_component_scores": {
            name: float(value)
            for name, value in zip(
                ATTENTION_RELATION_COMPONENT_NAMES,
                watermarked_carrier_paired_components.detach().cpu().tolist(),
            )
        },
        "final_image_attention_carrier_paired_component_gains": {
            name: float(value)
            for name, value in zip(
                ATTENTION_RELATION_COMPONENT_NAMES,
                carrier_paired_component_gains.detach().cpu().tolist(),
            )
        },
        "observability_status": "measured_from_carrier_only_counterfactual_real_qk",
    }


class _FullLatentSpace:
    """在 Null Space 消融中提供不改变方向的完整空间投影。"""

    solver_digest = "full_latent_space_ablation"

    @staticmethod
    def project(tensor: Any) -> Any:
        """原样返回 tensor。"""

        return tensor


def _encode_image_latent(pipeline: Any, image: Any) -> Any:
    """仅从待检图像执行 VAE 编码, 不读取生成轨迹。"""

    import torch

    dtype = next(pipeline.vae.parameters()).dtype
    pixels = pipeline.image_processor.preprocess(image).to(device=pipeline._execution_device, dtype=dtype)
    with torch.no_grad():
        encoded = pipeline.vae.encode(pixels).latent_dist.mode()
    shift_factor = float(getattr(pipeline.vae.config, "shift_factor", 0.0) or 0.0)
    scaling_factor = float(getattr(pipeline.vae.config, "scaling_factor", 1.0))
    return (encoded - shift_factor) * scaling_factor


def _public_detection_noise_seed(config: SemanticWatermarkRuntimeConfig) -> int:
    """由公开模型和检测协议派生与生成样本无关的固定噪声种子。"""

    detection_index = config.injection_step_indices[0] + 1
    payload = (
        f"slm_wm_image_only_attention|{config.carrier_model_reference}|{config.width}x{config.height}|"
        f"{config.inference_steps}|post_step_schedule_index={detection_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _image_attention_extractor(
    pipeline: Any,
    config: SemanticWatermarkRuntimeConfig,
    modules: tuple[tuple[str, Any], ...],
    prompt_embeds: Any,
    pooled_prompt_embeds: Any,
) -> Any:
    """构造只使用图像 VAE latent 的确定性注意力提取器。"""

    import torch

    detection_index = config.injection_step_indices[0] + 1
    public_detection_seed = _public_detection_noise_seed(config)

    def extract(image: Any) -> tuple[tuple[str, Any, tuple[int, ...]], ...]:
        """从任意待检图像提取全部冻结层的公开固定噪声 Q/K 关系。"""

        # img2img, 反演和再生成攻击会与主 pipeline 共享 scheduler 实例并改写
        # timesteps, begin_index 和 step_index. 每次检测都重新建立正式检测日程,
        # 保证 scale_noise 使用的 sigma 与 Transformer 前向 timestep 属于同一日程.
        pipeline.scheduler.set_timesteps(
            config.inference_steps,
            device=pipeline._execution_device,
        )
        timestep = pipeline.scheduler.timesteps[detection_index]
        scale_noise = getattr(pipeline.scheduler, "scale_noise", None)
        if not callable(scale_noise):
            raise RuntimeError(
                "正式仅图像 Q/K 提取要求 scheduler 提供可调用的 scale_noise"
            )
        latent = _encode_image_latent(pipeline, image)
        generator = torch.Generator(device=latent.device.type).manual_seed(public_detection_seed)
        noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
        timestep_batch = timestep.reshape(1).expand(latent.shape[0])
        noisy_latent = scale_noise(latent, timestep_batch, noise)
        with DifferentiableAttentionRecorder(modules, max_tokens=config.max_attention_tokens) as recorder:
            with torch.no_grad():
                _transformer_forward_function(
                    pipeline,
                    timestep,
                    prompt_embeds,
                    pooled_prompt_embeds,
                )(noisy_latent)
            if not recorder.records:
                raise RuntimeError("图像盲检没有捕获到真实 Q/K attention")
            records = tuple(
                (layer_name, attention.detach(), token_indices)
                for layer_name, attention, token_indices in recorder.records
            )
        return records

    return extract


def _align_image(image: Any, alignment: Any) -> Any:
    """依据恢复的仿射参考系对待检图像执行可复现重采样。"""

    import torch
    import torch.nn.functional as functional
    from PIL import Image

    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).reshape(height, width, 3)
    tensor = pixels.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    theta = torch.tensor(alignment.affine_transform, dtype=tensor.dtype).unsqueeze(0)
    grid = functional.affine_grid(
        theta,
        tensor.shape,
        align_corners=ATTENTION_GRID_ALIGN_CORNERS,
    )
    aligned = functional.grid_sample(
        tensor,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=ATTENTION_GRID_ALIGN_CORNERS,
    )
    output = (aligned[0].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).byte().numpy()
    return Image.fromarray(output, mode="RGB")


def _carrier_only_counterfactual_identity(
    full_config: SemanticWatermarkRuntimeConfig,
    carrier_only_config: SemanticWatermarkRuntimeConfig,
    full_update_records: list[dict[str, Any]],
    carrier_only_update_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """核验仅关闭 attention geometry 的总机制效应反事实身份。"""

    full_payload = semantic_watermark_runtime_config_payload(full_config)
    carrier_payload = semantic_watermark_runtime_config_payload(
        carrier_only_config
    )
    changed_fields = tuple(
        sorted(
            field_name
            for field_name in full_payload
            if full_payload[field_name] != carrier_payload[field_name]
        )
    )
    if changed_fields != ("attention_geometry_enabled",):
        raise RuntimeError(
            "carrier-only 反事实必须只关闭 attention_geometry_enabled"
        )
    if (
        full_config.attention_geometry_enabled is not True
        or carrier_only_config.attention_geometry_enabled is not False
        or full_config.seed != carrier_only_config.seed
    ):
        raise RuntimeError("carrier-only 反事实的 attention 开关或生成种子不一致")

    expected_steps = tuple(int(value) for value in full_config.injection_step_indices)
    expected_full_branches = _active_carrier_branch_names(full_config)
    expected_carrier_branches = _active_carrier_branch_names(
        carrier_only_config
    )
    if (
        len(full_update_records) != len(expected_steps)
        or len(carrier_only_update_records) != len(expected_steps)
    ):
        raise RuntimeError("完整方法与 carrier-only 必须精确覆盖全部注入步")

    required_content_sha256_fields = (
        "latent_content_sha256_before",
        "latent_content_sha256_after",
        "combined_update_content_sha256",
        "quantized_write_update_content_sha256",
        "adjacent_step_reference_latent_content_sha256",
        "lf_update_content_sha256",
        "tail_robust_update_content_sha256",
        "attention_geometry_update_content_sha256",
    )

    def validate_common_record(
        record: Mapping[str, Any],
        *,
        execution_role: str,
        attention_enabled: bool,
        expected_branches: tuple[str, ...],
    ) -> None:
        """验证反事实两侧更新原子的共同内容与分支身份。"""

        metadata = record.get("metadata")
        null_space_records = record.get("null_space_records")
        if not isinstance(metadata, Mapping) or not isinstance(
            null_space_records,
            Mapping,
        ):
            raise RuntimeError("反事实更新原子缺少 metadata 或 Null Space 记录")
        if (
            metadata.get("injection_execution_role") != execution_role
            or metadata.get("attention_geometry_enabled") is not attention_enabled
            or record.get("active_carrier_branches") != list(expected_branches)
            or set(null_space_records) != set(expected_branches)
        ):
            raise RuntimeError("反事实更新原子的执行角色或活动分支身份不一致")
        if record.get("tensor_content_digest_version") != (
            TENSOR_CONTENT_DIGEST_VERSION
        ):
            raise RuntimeError("反事实更新原子的 Tensor 内容摘要版本无效")
        for field_name in required_content_sha256_fields:
            value = str(record.get(field_name, ""))
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise RuntimeError("反事实更新原子缺少完整 tensor 内容 SHA-256")
        branch_update_content_records = {
            "lf_content": record.get("lf_update_content_sha256"),
            "tail_robust": record.get("tail_robust_update_content_sha256"),
            "attention_geometry": record.get(
                "attention_geometry_update_content_sha256"
            ),
        }
        if record.get("branch_updates_content_digest") != build_stable_digest(
            branch_update_content_records
        ):
            raise RuntimeError("反事实更新原子的三分支 Tensor 摘要不一致")
        branch_risk_records = record.get("branch_risk_records")
        if not isinstance(branch_risk_records, Mapping) or set(
            branch_risk_records
        ) != set(BRANCH_NAMES):
            raise RuntimeError("反事实更新原子缺少三个分支风险记录")
        branch_risk_content_records = {
            name: {
                field_name: branch_record.get(field_name)
                for field_name in (
                    "risk_values_content_sha256",
                    "budget_values_content_sha256",
                    "eligible_mask_content_sha256",
                )
            }
            for name, branch_record in branch_risk_records.items()
        }
        if (
            any(
                not _is_sha256_hex(value)
                for content_record in branch_risk_content_records.values()
                for value in content_record.values()
            )
            or record.get("branch_risk_content_digest")
            != build_stable_digest(branch_risk_content_records)
        ):
            raise RuntimeError("反事实更新原子的分支风险 Tensor 摘要不一致")
        step_index = record.get("step_index")
        if (
            not isinstance(step_index, int)
            or record.get("adjacent_step_reference_index") != step_index - 1
            or record.get("adjacent_step_stability_status")
            != "measured_from_immediately_previous_scheduler_step"
        ):
            raise RuntimeError("反事实更新原子的相邻调度步稳定度身份无效")
        expected_prg_digest = keyed_prg_protocol_record(
            full_config.keyed_prg_version
        )["keyed_prg_protocol_digest"]
        if (
            record.get("keyed_prg_version") != full_config.keyed_prg_version
            or record.get("keyed_prg_protocol_digest")
            != expected_prg_digest
        ):
            raise RuntimeError("反事实更新原子的密钥 PRG 协议身份无效")
        if (
            record.get("attention_module_names")
            != list(full_config.attention_module_names)
            or record.get("attention_coordinate_convention")
            != ATTENTION_COORDINATE_CONVENTION
            or record.get("attention_grid_align_corners")
            is not ATTENTION_GRID_ALIGN_CORNERS
        ):
            raise RuntimeError("反事实更新原子的注意力层或坐标身份无效")
        quantized_gate_applicable = record.get(
            "quantized_write_jacobian_gate_applicable"
        )
        quantized_gate_ready = record.get(
            "quantized_write_jacobian_gate_ready"
        )
        if full_config.null_space_enabled:
            relative_response = record.get(
                "quantized_write_relative_jacobian_response"
            )
            if (
                quantized_gate_applicable is not True
                or quantized_gate_ready is not True
                or not isinstance(relative_response, (int, float))
                or not math.isfinite(float(relative_response))
                or float(relative_response)
                > full_config.maximum_quantized_write_relative_jacobian_response
            ):
                raise RuntimeError("反事实更新原子的实际量化写回 Jacobian 门禁无效")
        elif (
            quantized_gate_applicable is not False
            or quantized_gate_ready is not False
            or record.get("quantized_write_jacobian_status")
            != "not_applicable_jacobian_null_space_disabled"
        ):
            raise RuntimeError("Null Space 消融错误声明实际量化写回 Jacobian 门禁")

    for full_record in full_update_records:
        validate_common_record(
            full_record,
            execution_role="full_method",
            attention_enabled=True,
            expected_branches=expected_full_branches,
        )
        if full_record.get("metadata", {}).get("attention_source") != (
            "real_qk_projection"
        ):
            raise RuntimeError("完整方法更新原子缺少真实 Q/K attention 来源")
        component_protocol = attention_relation_component_protocol(
            full_config.attention_relation_component_weights
        )
        if (
            full_record.get("attention_relation_component_names")
            != list(ATTENTION_RELATION_COMPONENT_NAMES)
            or full_record.get("attention_relation_active_component_names")
            != list(
                component_protocol[
                    "attention_relation_active_component_names"
                ]
            )
            or full_record.get("attention_relation_component_weights")
            != list(full_config.attention_relation_component_weights)
            or full_record.get(
                "attention_relation_component_protocol_digest"
            )
            != component_protocol[
                "attention_relation_component_protocol_digest"
            ]
        ):
            raise RuntimeError("完整方法更新原子的四分量权重协议无效")
        if (
            full_record.get("attention_qk_atomic_content_ready") is not True
            or not qk_atomic_evaluation_records_ready(
                full_record.get("attention_qk_atomic_content_records"),
                full_record.get("attention_qk_atomic_content_digest"),
                aggregate_field_name="attention_qk_atomic_content_records",
                expected_roles=(
                    "latent_before",
                    "content_base_latent",
                    "accepted_attention_candidate",
                    "actual_written_combined_latent",
                ),
                expected_layer_names=full_config.attention_module_names,
            )
        ):
            raise RuntimeError("完整方法更新原子缺少真实 Q/K 原子内容摘要")

    carrier_none_fields = (
        "attention_score_before",
        "attention_content_base_score",
        "attention_score_after",
        "attention_final_combined_score",
        "attention_score_gain",
        "attention_applied_update_strength",
        "attention_backtracking_step_count",
    )
    carrier_empty_string_fields = (
        "attention_update_digest",
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
    carrier_empty_list_fields = (
        "stable_token_indices",
        "attention_relation_component_names",
        "attention_relation_active_component_names",
        "attention_relation_component_weights",
        "attention_relation_qk_operator_metadata_records",
        "attention_qk_atomic_content_records",
    )
    for carrier_record in carrier_only_update_records:
        validate_common_record(
            carrier_record,
            execution_role="carrier_only_counterfactual",
            attention_enabled=False,
            expected_branches=expected_carrier_branches,
        )
        carrier_metadata = carrier_record["metadata"]
        if carrier_metadata.get("attention_source") != (
            "disabled_attention_geometry"
        ):
            raise RuntimeError("carrier-only 更新原子错误声明真实 Q/K attention 来源")
        if any(carrier_record.get(field_name) is not None for field_name in carrier_none_fields):
            raise RuntimeError("carrier-only 更新原子仍包含 attention 数值")
        if any(carrier_record.get(field_name) != "" for field_name in carrier_empty_string_fields):
            raise RuntimeError("carrier-only 更新原子仍包含 attention 或 pair 身份")
        if any(carrier_record.get(field_name) != [] for field_name in carrier_empty_list_fields):
            raise RuntimeError("carrier-only 更新原子仍包含 attention 关系集合")
        if carrier_record.get("attention_relation_direct_qk_source_ready") is not False:
            raise RuntimeError("carrier-only 更新原子错误声明直接 Q/K 来源")
        if carrier_record.get("attention_relation_qk_operator_metadata_ready") is not False:
            raise RuntimeError("carrier-only 更新原子错误声明 Q/K 算子元数据完整")
        if carrier_record.get("attention_qk_atomic_content_ready") is not False:
            raise RuntimeError("carrier-only 更新原子错误声明 Q/K 原子内容完整")

    def scheduler_trace(records: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        """提取足以核验相同 scheduler 轨迹的冻结字段。"""

        return tuple(
            {
                "step_index": int(record["step_index"]),
                "scheduler_step_timestep": float(
                    record["scheduler_step_timestep"]
                ),
                "post_step_schedule_index": int(
                    record["post_step_schedule_index"]
                ),
                "method_timestep": float(record["timestep"]),
            }
            for record in records
        )

    full_trace = scheduler_trace(full_update_records)
    carrier_trace = scheduler_trace(carrier_only_update_records)
    if (
        tuple(item["step_index"] for item in full_trace) != expected_steps
        or tuple(item["step_index"] for item in carrier_trace) != expected_steps
        or full_trace != carrier_trace
    ):
        raise RuntimeError("carrier-only 反事实没有复现完整方法的 scheduler 轨迹")
    full_initial_latent_sha256 = str(
        full_update_records[0]["latent_content_sha256_before"]
    )
    carrier_initial_latent_sha256 = str(
        carrier_only_update_records[0]["latent_content_sha256_before"]
    )
    if full_initial_latent_sha256 != carrier_initial_latent_sha256:
        raise RuntimeError("完整方法与 carrier-only 的首个注入前 latent 不一致")

    record = {
        "carrier_only_counterfactual_changed_fields": list(changed_fields),
        "carrier_only_counterfactual_generation_seed_random": int(
            full_config.seed
        ),
        "carrier_only_counterfactual_config_digest": build_stable_digest(
            carrier_payload
        ),
        "full_method_counterfactual_update_count": len(full_update_records),
        "carrier_only_counterfactual_update_count": len(
            carrier_only_update_records
        ),
        "full_method_counterfactual_update_records_digest": (
            build_stable_digest(full_update_records)
        ),
        "carrier_only_counterfactual_update_records_digest": (
            build_stable_digest(carrier_only_update_records)
        ),
        "carrier_only_counterfactual_atom_content_digest": (
            build_stable_digest(carrier_only_update_records)
        ),
        "full_method_initial_latent_content_sha256": (
            full_initial_latent_sha256
        ),
        "carrier_only_initial_latent_content_sha256": (
            carrier_initial_latent_sha256
        ),
        "carrier_only_counterfactual_initial_latent_identity_ready": True,
        "carrier_only_counterfactual_scheduler_trace": list(full_trace),
        "carrier_only_counterfactual_scheduler_trace_digest": (
            build_stable_digest(full_trace)
        ),
        "carrier_only_counterfactual_scheduler_identity_ready": True,
        "carrier_only_counterfactual_attention_geometry_enabled": False,
        "full_method_counterfactual_carrier_branches": list(
            expected_full_branches
        ),
        "carrier_only_counterfactual_carrier_branches": list(
            expected_carrier_branches
        ),
        "carrier_only_counterfactual_effect_scope": (
            "attention_geometry_switch_total_mechanism_effect"
        ),
        "carrier_only_counterfactual_realized_carrier_equality_assumed": False,
        "carrier_only_counterfactual_downstream_interactions_included": True,
    }
    record["carrier_only_counterfactual_identity_digest"] = build_stable_digest(
        record
    )
    record["carrier_only_counterfactual_ready"] = True
    return record


def run_semantic_watermark_runtime(
    config: SemanticWatermarkRuntimeConfig,
    runtime_context: SemanticWatermarkRuntimeContext | None = None,
) -> tuple[
    SemanticWatermarkRuntimeResult,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    Any,
    Any,
    Any | None,
    dict[str, Any],
]:
    """执行 clean/watermarked 生成和最终图像盲检。"""

    import torch

    started_at = time.time()
    run_id = build_semantic_watermark_run_id(config)
    context = runtime_context or load_semantic_watermark_runtime_context(config)
    pipeline = context.pipeline
    runtime_versions = context.runtime_versions
    feature_runtime = context.feature_runtime
    attention_modules = context.attention_modules
    unconditional_prompt = context.unconditional_prompt
    unconditional_pooled = context.unconditional_pooled
    diffusion_attack_runtime = context.diffusion_attack_runtime

    common_kwargs = {
        "prompt": config.prompt,
        "negative_prompt": config.negative_prompt,
        "width": config.width,
        "height": config.height,
        "num_inference_steps": config.inference_steps,
        "guidance_scale": config.guidance_scale,
        "output_type": "pil",
    }
    clean_generator = torch.Generator(device=config.device_name).manual_seed(config.seed)
    watermarked_generator = torch.Generator(device=config.device_name).manual_seed(config.seed)
    with torch.no_grad():
        clean_image = pipeline(generator=clean_generator, **common_kwargs).images[0]

    update_records: list[dict[str, Any]] = []
    active_update_records = update_records
    active_injection_config = config
    injection_execution_role = "full_method"
    previous_step_latent: Any | None = None

    def inject(pipe: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        nonlocal previous_step_latent
        latent = callback_kwargs.get("latents")
        if latent is None:
            return callback_kwargs
        if step_index not in active_injection_config.injection_step_indices:
            previous_step_latent = latent.detach().clone()
            return callback_kwargs
        if previous_step_latent is None:
            raise RuntimeError(
                "分支风险缺少紧邻上一 scheduler 步的真实 latent"
            )
        adjacent_step_reference_sha256 = tensor_content_sha256(
            previous_step_latent
        )
        post_step_index = step_index + 1
        scheduler_timesteps = pipe.scheduler.timesteps
        if post_step_index >= len(scheduler_timesteps):
            raise RuntimeError("post-step 注入缺少合法的下一调度时刻")
        method_timestep = scheduler_timesteps[post_step_index]
        with torch.enable_grad():
            signals = feature_runtime.branch_signal_maps(
                latent.float(),
                previous_step_latent.float(),
            )
            lf_template = build_low_frequency_template(
                latent,
                active_injection_config.key_material,
                active_injection_config.carrier_model_reference,
                prg_version=active_injection_config.keyed_prg_version,
            )
            tail_template, tail_threshold, retained_fraction = build_tail_robust_template(
                latent,
                active_injection_config.key_material,
                active_injection_config.carrier_model_reference,
                active_injection_config.tail_fraction if active_injection_config.tail_truncation_enabled else 1.0,
                prg_version=active_injection_config.keyed_prg_version,
            )
            transformer_forward = _transformer_forward_function(
                pipeline,
                method_timestep,
                unconditional_prompt,
                unconditional_pooled,
            )
            with DifferentiableAttentionRecorder(
                attention_modules,
                max_tokens=active_injection_config.max_attention_tokens,
            ) as recorder:
                attention_gradient = (
                    compute_attention_geometry_gradient(
                        latent,
                        transformer_forward,
                        recorder,
                        active_injection_config.key_material,
                        stable_token_fraction=(
                            active_injection_config.attention_stable_token_fraction
                        ),
                        unstable_pair_weight=(
                            active_injection_config.attention_unstable_pair_weight
                        ),
                        component_weights=(
                            active_injection_config.attention_relation_component_weights
                        ),
                    )
                    if active_injection_config.attention_geometry_enabled
                    else None
                )
                if attention_gradient is None:
                    recorder.clear()
                    with torch.no_grad():
                        transformer_forward(latent.detach().float())
                attention_stability = attention_relation_stability_map(
                    recorder.records,
                    tuple(int(value) for value in latent.shape[-2:]),
                ).detach()
                risk_bundle = build_branch_risk_fields(
                    semantic_values=signals["semantic"].mean(dim=0).reshape(-1).cpu().tolist(),
                    texture_values=signals["texture"].mean(dim=0).reshape(-1).cpu().tolist(),
                    adjacent_step_stability_values=signals[
                        "adjacent_step_stability"
                    ].mean(dim=0).reshape(-1).cpu().tolist(),
                    local_contrast_risk_values=signals[
                        "local_contrast_risk"
                    ].mean(dim=0).reshape(-1).cpu().tolist(),
                    attention_stability_values=(
                        attention_stability.mean(dim=0)
                        .reshape(-1)
                        .cpu()
                        .tolist()
                    ),
                    configs=_branch_risk_configs(active_injection_config),
                    required_eligible_branches=_required_branch_risk_eligibility(
                        active_injection_config
                    ),
                )
                branch_fields = {
                    "lf_content": risk_bundle.lf_content,
                    "tail_robust": risk_bundle.tail_robust,
                    "attention_geometry": risk_bundle.attention_geometry,
                }
                active_branch_names = _active_carrier_branch_names(
                    active_injection_config
                )
                active_branch_name_set = set(active_branch_names)
                active_branch_fields = {
                    branch_name: branch_field
                    for branch_name, branch_field in branch_fields.items()
                    if branch_name in active_branch_name_set
                }
                preferred_directions = {
                    "lf_content": (lf_template,),
                    "tail_robust": (tail_template,),
                    "attention_geometry": (
                        () if attention_gradient is None else (attention_gradient.gradient,)
                    ),
                }
                if active_injection_config.null_space_enabled:
                    linearized_latent = latent.float()
                    joint_feature_linearization = build_exact_jacobian_linearization(
                        feature_runtime.full_joint_feature_vector,
                        linearized_latent,
                    )
                    subspaces = {
                        branch_name: _solve_branch_subspace(
                            linearized_latent,
                            feature_runtime,
                            active_injection_config.key_material,
                            branch_name,
                            (
                                _branch_budget(signals["semantic"], branch_field)
                                if active_injection_config.semantic_routing_enabled
                                else tuple(1.0 for _ in branch_field.budget_values)
                            ),
                            active_injection_config.candidate_count,
                            active_injection_config.null_rank,
                            joint_feature_linearization,
                            preferred_directions[branch_name],
                            active_injection_config.maximum_relative_response_residual,
                            active_injection_config.minimum_projection_energy_retention,
                            active_injection_config.null_space_cg_max_iterations,
                            active_injection_config.null_space_cg_relative_tolerance,
                            active_injection_config.keyed_prg_version,
                        )
                        for branch_name, branch_field in active_branch_fields.items()
                    }
                    # Null Space 基底已经物化; 立即释放 JVP/VJP 图, 为 Q/K 回溯腾出显存。
                    del joint_feature_linearization
                else:
                    subspaces = {branch_name: _FullLatentSpace() for branch_name in active_branch_fields}
                jvp_modes = tuple(
                    sorted(
                        {
                            str(result.metadata.get("jvp_mode"))
                            for result in subspaces.values()
                            if hasattr(result, "metadata") and result.metadata.get("jvp_mode")
                        }
                    )
                )
                lf_carrier = (
                    project_canonical_template(
                        "lf_content",
                        lf_template,
                        subspaces["lf_content"],
                        active_injection_config.minimum_projection_energy_retention,
                        prg_version=active_injection_config.keyed_prg_version,
                    )
                    if active_injection_config.lf_enabled
                    else None
                )
                tail_carrier = (
                    project_canonical_template(
                        "tail_robust",
                        tail_template,
                        subspaces["tail_robust"],
                        active_injection_config.minimum_projection_energy_retention,
                        prg_version=active_injection_config.keyed_prg_version,
                    )
                    if active_injection_config.tail_robust_enabled
                    else None
                )
                latent_norm = latent.detach().float().norm()
                lf_update = (
                    lf_carrier.scaled_update(active_injection_config.lf_relative_strength * float(latent_norm.item()))
                    if lf_carrier is not None
                    else torch.zeros_like(latent)
                )
                tail_update = (
                    tail_carrier.scaled_update(active_injection_config.tail_relative_strength * float(latent_norm.item()))
                    if tail_carrier is not None
                    else torch.zeros_like(latent)
                )
                if attention_gradient is not None:
                    content_base_update = lf_update + tail_update
                    attention_update = optimize_attention_geometry_update(
                        latent=latent,
                        transformer_forward=transformer_forward,
                        recorder=recorder,
                        key_material=active_injection_config.key_material,
                        safe_subspace=subspaces["attention_geometry"],
                        update_strength=active_injection_config.attention_relative_strength * float(latent_norm.item()),
                        precomputed_gradient=attention_gradient,
                        base_update=content_base_update,
                        stable_token_fraction=(
                            active_injection_config.attention_stable_token_fraction
                        ),
                        unstable_pair_weight=(
                            active_injection_config.attention_unstable_pair_weight
                        ),
                        component_weights=(
                            active_injection_config.attention_relation_component_weights
                        ),
                    )
                    attention_tensor = attention_update.update
                    combined_update = content_base_update + attention_tensor
                    injected = latent + combined_update.to(dtype=latent.dtype)
                    recorder.clear()
                    with torch.no_grad():
                        transformer_forward(injected.detach().float())
                        final_score_tensor = attention_geometry_score(
                            recorder.records,
                            active_injection_config.key_material,
                            stable_pair_weights=(
                                attention_gradient.stable_pair_weights
                            ),
                            component_weights=(
                                active_injection_config.attention_relation_component_weights
                            ),
                        )
                        written_qk_identity = (
                            build_attention_relation_graph_identity(
                                recorder.records,
                                active_injection_config.key_material,
                                active_injection_config.attention_relation_component_weights,
                            )
                        )
                    final_score = float(final_score_tensor.detach().item())
                    required_score = max(
                        attention_update.score_before,
                        attention_update.content_base_score,
                    )
                    if not math.isfinite(final_score) or final_score <= required_score:
                        raise RuntimeError("真正写回的 combined latent 未提高真实 Q/K 目标")
                    if not written_qk_identity.qk_atomic_content_ready:
                        raise RuntimeError("真正写回的 combined latent 缺少 Q/K 原子摘要")
                    if (
                        written_qk_identity.component_protocol_digest
                        != attention_update.attention_relation_component_protocol_digest
                        or written_qk_identity.component_weights
                        != attention_update.attention_relation_component_weights
                    ):
                        raise RuntimeError("真正写回的 combined latent 四分量协议漂移")
                    qk_atomic_evaluation_records = (
                        *attention_update.qk_atomic_evaluation_records,
                        {
                            "qk_evaluation_role": "actual_written_combined_latent",
                            "qk_atomic_content_records": list(
                                written_qk_identity.qk_atomic_content_records
                            ),
                            "qk_atomic_content_digest": (
                                written_qk_identity.qk_atomic_content_digest
                            ),
                            "qk_atomic_content_ready": (
                                written_qk_identity.qk_atomic_content_ready
                            ),
                        },
                    )
                    attention_record = {
                        "attention_score_before": attention_update.score_before,
                        "attention_content_base_score": attention_update.content_base_score,
                        "attention_score_after": final_score,
                        "attention_final_combined_score": final_score,
                        "attention_score_gain": final_score - attention_update.score_before,
                        "attention_applied_update_strength": attention_update.applied_update_strength,
                        "attention_backtracking_step_count": attention_update.backtracking_step_count,
                        "attention_update_digest": attention_update.update_digest,
                        "stable_token_indices": list(
                            attention_update.stable_token_indices
                        ),
                        "stable_token_selection_digest": (
                            attention_update.stable_token_selection_digest
                        ),
                        "stable_pair_weight_identity_digest": (
                            attention_update.stable_pair_weight_identity_digest
                        ),
                        "stable_pair_weight_realization_digest": (
                            attention_update.stable_pair_weight_realization_digest
                        ),
                        "attention_relation_component_names": list(
                            attention_update.attention_relation_component_names
                        ),
                        "attention_relation_active_component_names": list(
                            attention_update.attention_relation_active_component_names
                        ),
                        "attention_relation_component_weights": list(
                            attention_update.attention_relation_component_weights
                        ),
                        "attention_relation_component_protocol_digest": (
                            attention_update.attention_relation_component_protocol_digest
                        ),
                        "attention_relation_source": (
                            attention_update.attention_relation_source
                        ),
                        "attention_relation_direct_qk_source_ready": (
                            attention_update.attention_relation_source
                            == DIRECT_QK_RELATION_SOURCE
                        ),
                        "attention_relation_probability_scope": (
                            "sampled_image_token_qk_relation_probability"
                        ),
                        "attention_relation_component_identity_digest": (
                            attention_update.attention_relation_component_identity_digest
                        ),
                        "attention_relation_keyed_projection_digest": (
                            attention_update.attention_relation_keyed_projection_digest
                        ),
                        "attention_relation_qk_operator_metadata_records": list(
                            attention_update.attention_relation_qk_operator_metadata_records
                        ),
                        "attention_relation_qk_operator_metadata_digest": (
                            attention_update.attention_relation_qk_operator_metadata_digest
                        ),
                        "attention_relation_qk_operator_metadata_ready": (
                            attention_update.attention_relation_qk_operator_metadata_ready
                        ),
                        "attention_qk_atomic_content_records": list(
                            qk_atomic_evaluation_records
                        ),
                        "attention_qk_atomic_content_digest": (
                            qk_atomic_evaluation_records_digest(
                                qk_atomic_evaluation_records,
                                "attention_qk_atomic_content_records",
                            )
                        ),
                        "attention_qk_atomic_content_ready": all(
                            bool(record["qk_atomic_content_ready"])
                            for record in qk_atomic_evaluation_records
                        ),
                    }
                else:
                    attention_tensor = torch.zeros_like(latent)
                    combined_update = lf_update + tail_update
                    injected = latent + combined_update.to(dtype=latent.dtype)
                    attention_record = {
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
                        "attention_relation_active_component_names": [],
                        "attention_relation_component_weights": [],
                        "attention_relation_component_protocol_digest": "",
                        "attention_relation_source": "",
                        "attention_relation_direct_qk_source_ready": False,
                        "attention_relation_probability_scope": "",
                        "attention_relation_component_identity_digest": "",
                        "attention_relation_keyed_projection_digest": "",
                        "attention_relation_qk_operator_metadata_records": [],
                        "attention_relation_qk_operator_metadata_digest": "",
                        "attention_relation_qk_operator_metadata_ready": False,
                        "attention_qk_atomic_content_records": [],
                        "attention_qk_atomic_content_digest": "",
                        "attention_qk_atomic_content_ready": False,
                    }
                quantized_write_jacobian_record = (
                    _quantized_write_jacobian_response_record(
                        (
                            feature_runtime.full_joint_feature_vector
                            if active_injection_config.null_space_enabled
                            else None
                        ),
                        latent,
                        injected,
                        active_injection_config.maximum_quantized_write_relative_jacobian_response,
                    )
                )
                if (
                    active_injection_config.null_space_enabled
                    and not quantized_write_jacobian_record[
                        "quantized_write_jacobian_gate_ready"
                    ]
                ):
                    raise RuntimeError(
                        "实际量化写回 Tensor 的完整 Jacobian 相对响应超过正式门禁"
                    )
                preservation_record = _combined_update_preservation_record(
                    feature_runtime,
                    latent,
                    injected,
                    config,
                )
                if (
                    active_injection_config.null_space_enabled
                    and not preservation_record["semantic_preservation_gate_ready"]
                ):
                    raise RuntimeError(
                        "真正写回的 combined latent 未通过完整语义与视觉保持门禁"
                    )
        branch_update_content_records = {
            "lf_content": tensor_content_sha256(lf_update),
            "tail_robust": tensor_content_sha256(tail_update),
            "attention_geometry": tensor_content_sha256(attention_tensor),
        }
        branch_risk_records = {
            name: _branch_risk_record(branch_field)
            for name, branch_field in branch_fields.items()
        }
        branch_risk_content_records = {
            name: {
                "risk_values_content_sha256": branch_record[
                    "risk_values_content_sha256"
                ],
                "budget_values_content_sha256": branch_record[
                    "budget_values_content_sha256"
                ],
                "eligible_mask_content_sha256": branch_record[
                    "eligible_mask_content_sha256"
                ],
            }
            for name, branch_record in branch_risk_records.items()
        }
        active_update_records.append(
            {
                "run_id": run_id,
                "prompt_id": active_injection_config.prompt_id,
                "split": active_injection_config.split,
                "step_index": int(step_index),
                "scheduler_step_timestep": float(timestep.detach().float().item()),
                "post_step_schedule_index": int(post_step_index),
                "adjacent_step_reference_index": int(step_index - 1),
                "adjacent_step_reference_latent_content_sha256": (
                    adjacent_step_reference_sha256
                ),
                "adjacent_step_stability_status": (
                    "measured_from_immediately_previous_scheduler_step"
                ),
                "timestep": float(method_timestep.detach().float().item()),
                "latent_content_sha256_before": tensor_content_sha256(latent),
                "latent_content_sha256_after": tensor_content_sha256(injected),
                "combined_update_content_sha256": tensor_content_sha256(
                    combined_update
                ),
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
                "tensor_content_digest_version": (
                    TENSOR_CONTENT_DIGEST_VERSION
                ),
                "relative_update_norm": tensor_norm(combined_update) / max(tensor_norm(latent), 1e-12),
                "active_carrier_branches": list(active_branch_names),
                "branch_risk_bundle_digest": risk_bundle.bundle_digest,
                "branch_risk_records": branch_risk_records,
                "branch_risk_content_digest": build_stable_digest(
                    branch_risk_content_records
                ),
                "null_space_records": {
                    name: (
                        result.to_record()
                        if hasattr(result, "to_record")
                        else {"branch_name": name, "solver": result.solver_digest}
                    )
                    for name, result in subspaces.items()
                },
                "lf_projection_energy_retention": (
                    None if lf_carrier is None else lf_carrier.projection_energy_retention
                ),
                "tail_projection_energy_retention": (
                    None if tail_carrier is None else tail_carrier.projection_energy_retention
                ),
                "tail_threshold": tail_threshold,
                "tail_retained_fraction": retained_fraction,
                "keyed_prg_version": active_injection_config.keyed_prg_version,
                "keyed_prg_protocol_digest": keyed_prg_protocol_record(
                    active_injection_config.keyed_prg_version
                )["keyed_prg_protocol_digest"],
                "attention_module_names": list(
                    active_injection_config.attention_module_names
                ),
                "attention_coordinate_convention": (
                    active_injection_config.attention_coordinate_convention
                ),
                "attention_grid_align_corners": (
                    active_injection_config.attention_grid_align_corners
                ),
                **attention_record,
                **quantized_write_jacobian_record,
                **preservation_record,
                "metadata": {
                    "jvp_mode": jvp_modes[0] if len(jvp_modes) == 1 else "disabled_or_mixed",
                    "jvp_modes": list(jvp_modes),
                    "basis_solver": "matrix_free_full_jacobian_psd_cg",
                    "attention_source": (
                        "real_qk_projection"
                        if active_injection_config.attention_geometry_enabled
                        else "disabled_attention_geometry"
                    ),
                    "detector_requires_generation_trace": False,
                    "semantic_routing_enabled": active_injection_config.semantic_routing_enabled,
                    "branch_risk_mode": active_injection_config.branch_risk_mode,
                    "null_space_enabled": active_injection_config.null_space_enabled,
                    "lf_enabled": active_injection_config.lf_enabled,
                    "tail_robust_enabled": active_injection_config.tail_robust_enabled,
                    "tail_truncation_enabled": active_injection_config.tail_truncation_enabled,
                    "attention_geometry_enabled": active_injection_config.attention_geometry_enabled,
                    "injection_execution_role": injection_execution_role,
                    "supports_paper_claim": False,
                },
            }
        )
        previous_step_latent = injected.detach().clone()
        callback_kwargs["latents"] = injected.detach().to(dtype=latent.dtype)
        return callback_kwargs

    watermarked_image = pipeline(
        generator=watermarked_generator,
        callback_on_step_end=inject,
        callback_on_step_end_tensor_inputs=["latents"],
        **common_kwargs,
    ).images[0]
    carrier_only_image: Any | None = None
    carrier_only_counterfactual: dict[str, Any] | None = None
    carrier_only_final_image_preservation: dict[str, Any] | None = None
    final_image_preservation: dict[str, Any] | None = None
    carrier_only_update_records: list[dict[str, Any]] = []
    if config.attention_geometry_enabled:
        carrier_only_config = replace(
            config,
            attention_geometry_enabled=False,
        )
        active_update_records = carrier_only_update_records
        active_injection_config = carrier_only_config
        injection_execution_role = "carrier_only_counterfactual"
        previous_step_latent = None
        carrier_only_generator = torch.Generator(
            device=config.device_name
        ).manual_seed(config.seed)
        carrier_only_image = pipeline(
            generator=carrier_only_generator,
            callback_on_step_end=inject,
            callback_on_step_end_tensor_inputs=["latents"],
            **common_kwargs,
        ).images[0]
        carrier_only_counterfactual = _carrier_only_counterfactual_identity(
            config,
            carrier_only_config,
            update_records,
            carrier_only_update_records,
        )
        active_update_records = update_records
        active_injection_config = config
        injection_execution_role = "full_method"
        previous_step_latent = None
        (
            final_image_preservation,
            carrier_only_final_image_preservation,
        ) = _three_way_final_image_preservation_records(
            pipeline,
            feature_runtime,
            clean_image,
            carrier_only_image,
            watermarked_image,
            config,
            carrier_only_counterfactual,
        )
        if not carrier_only_final_image_preservation[
            "carrier_only_counterfactual_three_way_preservation_gate_ready"
        ]:
            raise RuntimeError(
                "最终 clean、carrier-only 与完整方法成图未通过三边特征保持门禁"
            )
    attention_extractor = (
        _image_attention_extractor(
            pipeline,
            config,
            attention_modules,
            unconditional_prompt,
            unconditional_pooled,
        )
        if config.attention_geometry_enabled
        else None
    )
    final_image_attention_observability = (
        _final_image_attention_observability_record(
            attention_extractor,
            clean_image,
            carrier_only_image,
            watermarked_image,
            config,
            carrier_only_counterfactual=carrier_only_counterfactual,
            require_gpu_execution=True,
        )
    )
    if (
        config.attention_geometry_enabled
        and not final_image_attention_observability[
            "final_image_attention_observability_gate_ready"
        ]
    ):
        raise RuntimeError(
            "最终 carrier-only/完整方法成图未通过真实 Q/K 双归因门禁"
        )
    if final_image_preservation is None:
        final_image_preservation = _final_image_preservation_record(
            pipeline,
            feature_runtime,
            clean_image,
            watermarked_image,
            config,
        )
    if (
        config.null_space_enabled
        and not final_image_preservation["final_image_preservation_gate_ready"]
    ):
        raise RuntimeError("最终 clean/watermarked 成图未通过累计完整特征保持门禁")
    paired_quality = compute_image_quality_metrics(clean_image, watermarked_image)

    lf_weight = 0.70 if config.lf_enabled and config.tail_robust_enabled else (1.0 if config.lf_enabled else 0.0)
    tail_weight = 1.0 - lf_weight
    detector_config = ImageOnlyDetectionConfig(
        model_id=config.carrier_model_reference,
        keyed_prg_version=config.keyed_prg_version,
        content_threshold=config.content_threshold,
        geometry_score_threshold=config.geometry_score_threshold,
        registration_confidence_threshold=config.registration_confidence_threshold,
        attention_sync_score_threshold=config.attention_sync_score_threshold,
        rescue_margin_low=config.rescue_margin_low,
        lf_weight=lf_weight,
        tail_robust_weight=tail_weight,
        tail_fraction=config.tail_fraction if config.tail_truncation_enabled else 1.0,
        attention_stable_token_fraction=(
            config.attention_stable_token_fraction
        ),
        attention_unstable_pair_weight=(
            config.attention_unstable_pair_weight
        ),
        attention_relation_component_weights=(
            config.attention_relation_component_weights
        ),
    )
    def adversarial_detection_score(candidate: Any) -> float:
        """返回与最终内容主判和几何对齐救回一致的连续攻击目标。"""

        evaluated = detect_image_only_watermark(
            image=candidate,
            key_material=config.key_material,
            config=detector_config,
            image_latent_encoder=lambda image: _encode_image_latent(pipeline, image),
            image_attention_extractor=(
                attention_extractor if config.attention_geometry_enabled else None
            ),
            image_aligner=_align_image if config.image_alignment_enabled else None,
        )
        scores = [evaluated.content.content_score]
        if evaluated.geometry_reliable and evaluated.aligned_content_score is not None:
            scores.append(evaluated.aligned_content_score)
        return max(scores)

    detections = []
    for sample_role, image, detection_key in (
        ("clean_negative", clean_image, config.key_material),
        ("positive_source", watermarked_image, config.key_material),
        ("wrong_key_negative", watermarked_image, f"{config.key_material}|wrong-key"),
    ):
        detection = detect_image_only_watermark(
            image=image,
            key_material=detection_key,
            config=detector_config,
            image_latent_encoder=lambda candidate: _encode_image_latent(pipeline, candidate),
            image_attention_extractor=attention_extractor if config.attention_geometry_enabled else None,
            image_aligner=_align_image if config.image_alignment_enabled else None,
        )
        record = detection.to_record()
        record["run_id"] = run_id
        record["prompt_id"] = config.prompt_id
        record["split"] = config.split
        record["sample_role"] = sample_role
        record["metadata"] = {
            **record["metadata"],
            "supports_paper_claim": False,
            "threshold_status": "development_only_until_calibration_freeze",
        }
        detections.append(record)

    attacked_images: dict[str, Any] = {}
    attack_configs = tuple(
        attack
        for attack in default_attack_configs()
        if attack.enabled
        and not attack.requires_gpu
        and attack.resource_profile in set(config.standard_attack_profiles)
    )
    for sample_role, source_image in (("clean_negative", clean_image), ("positive_source", watermarked_image)):
        for attack_index, attack_config in enumerate(attack_configs):
            attacked_image = apply_standard_image_attack(
                source_image,
                attack_config,
                seed=config.seed + attack_index,
            )
            image_key = f"{sample_role}_{attack_config.attack_id}"
            attacked_images[image_key] = attacked_image
            detection = detect_image_only_watermark(
                image=attacked_image,
                key_material=config.key_material,
                config=detector_config,
                image_latent_encoder=lambda candidate: _encode_image_latent(pipeline, candidate),
                image_attention_extractor=attention_extractor if config.attention_geometry_enabled else None,
                image_aligner=_align_image if config.image_alignment_enabled else None,
            )
            record = detection.to_record()
            record.update(
                {
                    "run_id": run_id,
                    "prompt_id": config.prompt_id,
                    "split": config.split,
                    "sample_role": sample_role,
                    "attack_id": attack_config.attack_id,
                    "attack_family": attack_config.attack_family,
                    "attack_name": attack_config.attack_name,
                    "resource_profile": attack_config.resource_profile,
                    "attack_config_digest": attack_config_digest(attack_config),
                    "attack_parameters": attack_config.attack_parameters,
                    "attack_performed": True,
                    "attacked_image_key": image_key,
                }
            )
            record["metadata"] = {
                **record["metadata"],
                "metric_status": "measured_from_real_attacked_image",
                "supports_paper_claim": False,
                "threshold_status": "development_only_until_calibration_freeze",
            }
            detections.append(record)

    if config.diffusion_attacks_enabled:
        if diffusion_attack_runtime is None:
            raise RuntimeError("diffusion_attacks_enabled 要求共享再扩散攻击运行时")
        formal_attack_configs_by_id = {
            attack.attack_id: attack for attack in default_attack_configs()
        }
        for sample_role, source_image in (("clean_negative", clean_image), ("positive_source", watermarked_image)):
            for attack_index, attack_spec in enumerate(default_diffusion_attack_specs()):
                attack_execution = diffusion_attack_runtime.apply(
                    source_image,
                    attack_spec,
                    seed=config.seed + 20000 + attack_index,
                    prompt_text=config.prompt,
                    detection_score=adversarial_detection_score,
                )
                attacked_image = attack_execution.image
                image_key = f"{sample_role}_{attack_spec.attack_id}"
                attacked_images[image_key] = attacked_image
                detection = detect_image_only_watermark(
                    image=attacked_image,
                    key_material=config.key_material,
                    config=detector_config,
                    image_latent_encoder=lambda candidate: _encode_image_latent(pipeline, candidate),
                    image_attention_extractor=attention_extractor if config.attention_geometry_enabled else None,
                    image_aligner=_align_image if config.image_alignment_enabled else None,
                )
                record = detection.to_record()
                record.update(
                    {
                        "run_id": run_id,
                        "prompt_id": config.prompt_id,
                        "split": config.split,
                        "sample_role": sample_role,
                        "attack_id": attack_spec.attack_id,
                        "attack_family": attack_spec.attack_family,
                        "attack_name": attack_spec.attack_name,
                        "resource_profile": "full_extra",
                        "attack_config_digest": attack_config_digest(
                            formal_attack_configs_by_id[attack_spec.attack_id]
                        ),
                        "attack_parameters": attack_spec.attack_parameters,
                        "attack_implementation": attack_spec.attack_implementation,
                        "attack_execution": attack_execution.to_record(),
                        "attack_performed": True,
                        "attacked_image_key": image_key,
                    }
                )
                record["metadata"] = {
                    **record["metadata"],
                    "metric_status": "measured_from_real_diffusion_attacked_image",
                    "supports_paper_claim": False,
                    "threshold_status": "development_only_until_calibration_freeze",
                }
                detections.append(record)

    elapsed_seconds = time.time() - started_at
    random_identity_random = {
        "generation_seed_random": int(config.seed),
        "public_detection_seed_random": int(_public_detection_noise_seed(config)),
        "key_material_digest_random": build_stable_digest(
            {"key_material": config.key_material}
        ),
        "standard_attack_seeds_random": {
            attack.attack_id: int(config.seed + attack_index)
            for attack_index, attack in enumerate(attack_configs)
        },
        "diffusion_attack_seeds_random": {
            attack.attack_id: int(config.seed + 20000 + attack_index)
            for attack_index, attack in enumerate(default_diffusion_attack_specs())
        }
        if config.diffusion_attacks_enabled
        else {},
    }
    scientific_unit_provenance = build_scientific_unit_provenance(
        scientific_unit_id=run_id,
        scientific_unit_config_digest=semantic_watermark_runtime_config_digest(config),
        runtime_environment=runtime_versions["runtime_environment"],
        execution_device_name=str(pipeline._execution_device),
        torch_module=torch,
        random_identity_random=random_identity_random,
    )
    result = SemanticWatermarkRuntimeResult(
        run_id=run_id,
        run_decision="pass" if update_records else "fail",
        clean_image_path="",
        watermarked_image_path="",
        update_record_path="",
        detection_record_path="",
        manifest_path="",
        update_count=len(update_records),
        clean_detection_positive=bool(detections[0]["evidence_positive"]),
        watermarked_detection_positive=bool(detections[1]["evidence_positive"]),
        elapsed_seconds=elapsed_seconds,
        metadata={
            **runtime_versions,
            "method_runtime": "real_scientific_operators",
            "method_definition": semantic_conditioned_latent_method_definition(),
            "method_definition_digest": (
                semantic_conditioned_latent_method_definition_digest()
            ),
            "detector_input_access_mode": "image_key_public_model_only",
            "supports_paper_claim": False,
            "paired_quality": paired_quality,
            "final_image_preservation": final_image_preservation,
            "carrier_only_final_image_preservation": (
                carrier_only_final_image_preservation
            ),
            "carrier_only_counterfactual": carrier_only_counterfactual,
            "final_image_attention_observability": (
                final_image_attention_observability
            ),
            "scientific_unit_config": semantic_watermark_runtime_config_payload(
                config
            ),
            "scientific_unit_provenance": scientific_unit_provenance,
        },
    )
    return (
        result,
        tuple(update_records),
        tuple(carrier_only_update_records),
        tuple(detections),
        clean_image,
        watermarked_image,
        carrier_only_image,
        attacked_images,
    )


def write_semantic_watermark_runtime_outputs(
    config: SemanticWatermarkRuntimeConfig,
    root: str | Path = ".",
    runtime_context: SemanticWatermarkRuntimeContext | None = None,
) -> SemanticWatermarkRuntimeResult:
    """运行真实方法并把全部持久化产物写入 outputs。"""

    root_path = Path(root).resolve()
    output_dir = (root_path / config.output_dir).resolve()
    outputs_root = (root_path / "outputs").resolve()
    if output_dir != outputs_root and outputs_root not in output_dir.parents:
        raise ValueError("真实方法输出必须位于 outputs 目录")
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        result,
        update_records,
        carrier_only_update_records,
        detections,
        clean_image,
        watermarked_image,
        carrier_only_image,
        attacked_images,
    ) = run_semantic_watermark_runtime(
        config,
        runtime_context=runtime_context,
    )
    run_dir = output_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    clean_image_path = run_dir / "clean_image.png"
    watermarked_image_path = run_dir / "watermarked_image.png"
    carrier_only_image_path = run_dir / "carrier_only_image.png"
    clean_image.save(clean_image_path)
    watermarked_image.save(watermarked_image_path)
    if carrier_only_image is not None:
        carrier_only_image.save(carrier_only_image_path)
    attacked_image_dir = run_dir / "attacked_images"
    attacked_image_dir.mkdir(parents=True, exist_ok=True)
    attacked_image_paths = []
    attacked_image_path_by_key: dict[str, Path] = {}
    for image_key, attacked_image in sorted(attacked_images.items()):
        attacked_path = attacked_image_dir / f"{image_key}.png"
        attacked_image.save(attacked_path)
        attacked_image_path_by_key[image_key] = attacked_path
        attacked_image_paths.append(attacked_path.relative_to(root_path).as_posix())
    update_path = run_dir / "latent_update_records.jsonl"
    carrier_only_update_path = run_dir / "carrier_only_update_records.jsonl"
    detection_path = run_dir / "image_only_detection_records.jsonl"
    result_path = run_dir / "runtime_result.json"
    governed_detections = []
    for detection in detections:
        record = dict(detection)
        sample_role = str(record.get("sample_role", ""))
        source_path = clean_image_path if sample_role == "clean_negative" else watermarked_image_path
        source_image = clean_image if sample_role == "clean_negative" else watermarked_image
        attacked_image_key = str(record.get("attacked_image_key", ""))
        evaluated_path = attacked_image_path_by_key.get(attacked_image_key, source_path)
        evaluated_image = attacked_images.get(attacked_image_key, source_image)
        source_to_evaluated_quality = compute_image_quality_metrics(source_image, evaluated_image)
        record.update(
            {
                "run_id": result.run_id,
                "source_image_path": source_path.relative_to(root_path).as_posix(),
                "source_image_digest": file_digest(source_path),
                "evaluated_image_path": evaluated_path.relative_to(root_path).as_posix(),
                "evaluated_image_digest": file_digest(evaluated_path),
                "attacked_image_path": (
                    evaluated_path.relative_to(root_path).as_posix() if attacked_image_key else ""
                ),
                "attacked_image_digest": file_digest(evaluated_path) if attacked_image_key else "",
                "source_to_evaluated_ssim": float(source_to_evaluated_quality["ssim"]),
                "source_to_evaluated_psnr": source_to_evaluated_quality["psnr"],
                "source_to_evaluated_mse": float(source_to_evaluated_quality["mse"]),
            }
        )
        governed_detections.append(record)
    update_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in update_records), encoding="utf-8")
    if carrier_only_image is not None:
        carrier_only_update_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in carrier_only_update_records
            ),
            encoding="utf-8",
        )
    detection_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in governed_detections), encoding="utf-8")
    resolved_result_payload = result.to_dict()
    resolved_metadata = dict(resolved_result_payload["metadata"])
    observability_record = dict(
        resolved_metadata.get("final_image_attention_observability", {})
    )
    carrier_preservation_record = dict(
        resolved_metadata.get("carrier_only_final_image_preservation") or {}
    )
    counterfactual_record = dict(
        resolved_metadata.get("carrier_only_counterfactual") or {}
    )
    if carrier_only_image is not None:
        carrier_image_path = carrier_only_image_path.relative_to(root_path).as_posix()
        carrier_image_digest = file_digest(carrier_only_image_path)
        carrier_image_identity = {
            "carrier_only_counterfactual_image_path": carrier_image_path,
            "carrier_only_counterfactual_image_digest": carrier_image_digest,
        }
        carrier_atom_content_digest = build_stable_digest(
            list(carrier_only_update_records)
        )
        if (
            counterfactual_record.get(
                "carrier_only_counterfactual_atom_content_digest"
            )
            != carrier_atom_content_digest
        ):
            raise RuntimeError("carrier-only 持久化原子的内容摘要与运行身份不一致")
        carrier_atom_identity = {
            "carrier_only_counterfactual_atom_path": (
                carrier_only_update_path.relative_to(root_path).as_posix()
            ),
            "carrier_only_counterfactual_atom_file_sha256": file_digest(
                carrier_only_update_path
            ),
            "carrier_only_counterfactual_atom_content_digest": (
                carrier_atom_content_digest
            ),
        }
        observability_record.update(carrier_image_identity)
        observability_record.update(carrier_atom_identity)
        carrier_preservation_record.update(carrier_image_identity)
        carrier_preservation_record.update(carrier_atom_identity)
        counterfactual_record.update(carrier_image_identity)
        counterfactual_record.update(carrier_atom_identity)
    resolved_metadata["final_image_attention_observability"] = (
        observability_record
    )
    resolved_metadata["carrier_only_final_image_preservation"] = (
        carrier_preservation_record or None
    )
    resolved_metadata["carrier_only_counterfactual"] = (
        counterfactual_record or None
    )
    resolved_result = SemanticWatermarkRuntimeResult(
        **{
            **resolved_result_payload,
            "metadata": resolved_metadata,
            "clean_image_path": clean_image_path.relative_to(root_path).as_posix(),
            "watermarked_image_path": watermarked_image_path.relative_to(root_path).as_posix(),
            "update_record_path": update_path.relative_to(root_path).as_posix(),
            "detection_record_path": detection_path.relative_to(root_path).as_posix(),
            "manifest_path": (run_dir / "manifest.local.json").relative_to(root_path).as_posix(),
        }
    )
    result_path.write_text(_stable_json(resolved_result.to_dict()), encoding="utf-8")
    manifest_path = run_dir / "manifest.local.json"
    manifest = build_artifact_manifest(
        artifact_id=f"{result.run_id}_manifest",
        artifact_type="local_manifest",
        input_paths=(),
        output_paths=(
            update_path.relative_to(root_path).as_posix(),
            detection_path.relative_to(root_path).as_posix(),
            result_path.relative_to(root_path).as_posix(),
            clean_image_path.relative_to(root_path).as_posix(),
            watermarked_image_path.relative_to(root_path).as_posix(),
            *(
                (carrier_only_image_path.relative_to(root_path).as_posix(),)
                if carrier_only_image is not None
                else ()
            ),
            *(
                (carrier_only_update_path.relative_to(root_path).as_posix(),)
                if carrier_only_image is not None
                else ()
            ),
            manifest_path.relative_to(root_path).as_posix(),
            *attacked_image_paths,
        ),
        config=semantic_watermark_runtime_config_payload(config),
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 experiments.runners.semantic_watermark_runtime.write_semantic_watermark_runtime_outputs",
        metadata={
            "run_id": result.run_id,
            "protocol_decision": result.run_decision,
            "detector_input_access_mode": "image_key_public_model_only",
            "supports_paper_claim": False,
            **(
                {
                    "carrier_only_counterfactual_identity_digest": (
                        observability_record[
                            "carrier_only_counterfactual_identity_digest"
                        ]
                    ),
                    "carrier_only_counterfactual_image_digest": (
                        observability_record[
                            "carrier_only_counterfactual_image_digest"
                        ]
                    ),
                    "carrier_only_counterfactual_atom_file_sha256": (
                        counterfactual_record[
                            "carrier_only_counterfactual_atom_file_sha256"
                        ]
                    ),
                    "carrier_only_counterfactual_atom_path": (
                        counterfactual_record[
                            "carrier_only_counterfactual_atom_path"
                        ]
                    ),
                    "carrier_only_counterfactual_atom_content_digest": (
                        counterfactual_record[
                            "carrier_only_counterfactual_atom_content_digest"
                        ]
                    ),
                }
                if carrier_only_image is not None
                else {}
            ),
        },
    ).to_dict()
    manifest_path.write_text(_stable_json(manifest), encoding="utf-8")
    output_parts = Path(config.output_dir).parts
    checkpoint_roles = {
        "image_only_dataset_runtime": "image_only_dataset_runtime",
        "formal_mechanism_ablation": "runtime_rerun_ablation",
    }
    if (
        len(output_parts) >= 3
        and output_parts[0] == "outputs"
        and output_parts[1] in checkpoint_roles
    ):
        persist_completed_unit_from_manifest(
            manifest_path,
            repository_root=root_path,
            artifact_role=checkpoint_roles[output_parts[1]],
            paper_run_name=output_parts[2],
        )
    return resolved_result
