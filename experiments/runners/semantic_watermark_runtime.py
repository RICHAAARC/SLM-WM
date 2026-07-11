"""运行真实语义安全子空间嵌入和仅图像检测闭环。

该 runner 属于核心方法复现层, 在真实 SD3/SD3.5 latent 上计算分支风险、
JVP/SVD 低响应子空间、安全投影、真实 Q/K 注意力梯度和最终图像盲检。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

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
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from main.methods.carrier import (
    build_low_frequency_template,
    build_tail_robust_template,
    project_canonical_template,
)
from main.methods.detection import ImageOnlyDetectionConfig, detect_image_only_watermark
from main.methods.geometry import (
    DifferentiableAttentionRecorder,
    attention_relation_stability_map,
    compute_attention_geometry_gradient,
    optimize_attention_geometry_update,
)
from main.methods.semantic import build_branch_risk_fields
from main.methods.subspace import (
    ExactJVPLinearization,
    build_exact_jvp_linearization,
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
    tail_fraction: float = _FORMAL_METHOD_CONFIG.tail_fraction
    minimum_projection_energy_retention: float = _FORMAL_METHOD_CONFIG.minimum_projection_energy_retention
    maximum_relative_response_residual: float = _FORMAL_METHOD_CONFIG.maximum_relative_response_residual
    max_attention_tokens: int = _FORMAL_METHOD_CONFIG.max_attention_tokens
    attention_module_count: int = _FORMAL_METHOD_CONFIG.attention_module_count
    semantic_routing_enabled: bool = True
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
        if any(index < 0 or index >= self.inference_steps for index in self.injection_step_indices):
            raise ValueError("注入时刻必须位于推理步范围内")
        if not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction 必须位于 (0, 1]")
        if not 0.0 < self.minimum_projection_energy_retention <= 1.0:
            raise ValueError("minimum_projection_energy_retention 必须位于 (0, 1]")
        if not 0.0 < self.maximum_relative_response_residual <= 1.0:
            raise ValueError("maximum_relative_response_residual 必须位于 (0, 1]")
        if not self.lf_enabled and not self.tail_robust_enabled:
            raise ValueError("正式内容检测至少需要启用一个内容载体分支")
        if self.attention_module_count < 2:
            raise ValueError("真实注意力关系稳定度至少需要两个 Q/K 注意力层")
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
    attention_modules = _attention_modules(pipeline, config.attention_module_count)
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


def _tensor_digest(tensor: Any) -> str:
    """为真实 tensor 生成不保存原始值的稳定摘要。"""

    values = tensor.detach().float().cpu()
    return build_stable_digest(
        {
            "shape": tuple(int(value) for value in values.shape),
            "mean": round(float(values.mean().item()), 12),
            "std": round(float(values.std(unbiased=False).item()), 12),
            "norm": round(float(values.norm().item()), 12),
        }
    )


def build_semantic_watermark_run_id(config: SemanticWatermarkRuntimeConfig) -> str:
    """根据完整运行配置生成稳定标识。"""

    payload = asdict(config)
    payload["key_material"] = build_stable_digest({"key_material": config.key_material})
    return f"semantic_watermark_{build_stable_digest(payload)[:16]}"


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
    config_payload = {**asdict(config), "key_material": build_stable_digest({"key_material": config.key_material})}
    expected_config_digest = build_stable_digest(config_payload)
    if manifest.get("config_digest") != expected_config_digest:
        return None
    if manifest.get("code_version") != resolve_code_version(root_path):
        return None
    if result_payload.get("run_decision") != "pass":
        return None
    output_paths = tuple(str(path) for path in manifest.get("output_paths", ()))
    if not output_paths or not all((root_path / path).is_file() for path in output_paths):
        return None
    try:
        return SemanticWatermarkRuntimeResult(**result_payload)
    except TypeError:
        return None


def _attention_modules(pipeline: Any, limit: int) -> tuple[tuple[str, Any], ...]:
    """选择真实 Transformer 中公开 Q/K 投影的自注意力模块。"""

    candidates = []
    for name, module in pipeline.transformer.named_modules():
        if hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "heads"):
            candidates.append((name, module))
    if len(candidates) < limit:
        raise RuntimeError("模型中的真实 Q/K 注意力模块数量不足")
    if limit == 1:
        return (candidates[len(candidates) // 2],)
    selected = []
    for index in range(limit):
        selected_index = round(index * (len(candidates) - 1) / (limit - 1))
        if candidates[selected_index] not in selected:
            selected.append(candidates[selected_index])
    if len(selected) != limit:
        raise RuntimeError("无法选择足够数量且互不重复的真实 Q/K 注意力模块")
    return tuple(selected)


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


def _branch_risk_record(branch_field: Any) -> dict[str, Any]:
    """生成不保存大型空间数组的可审计分支风险摘要。"""

    return {
        "branch_name": branch_field.branch_name,
        "risk_field_digest": branch_field.risk_field_digest,
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
    joint_feature_linearization: ExactJVPLinearization,
    preferred_directions: tuple[Any, ...] = (),
    maximum_relative_response_residual: float = 1e-4,
) -> Any:
    """为一个载体分支运行真实 JVP 和 SVD。"""

    candidates = generate_keyed_candidate_directions(
        latent,
        key_material,
        branch_name,
        candidate_count,
        axis_budget,
        preferred_directions,
    )
    result = solve_jacobian_null_space(
        latent=latent.float(),
        semantic_feature_function=feature_runtime.semantic_condition_features,
        visual_feature_function=feature_runtime.visual_condition_features,
        candidate_matrix=candidates,
        null_rank=null_rank,
        visual_response_weight=1.0,
        branch_name=branch_name,
        joint_feature_function=feature_runtime.joint_condition_features,
        joint_feature_linearization=joint_feature_linearization,
    )
    result.metadata["preferred_direction_count"] = len(preferred_directions)
    result.metadata["preferred_direction_role"] = "carrier_or_attention_gradient"
    result.metadata["maximum_relative_response_residual"] = maximum_relative_response_residual
    if result.relative_response_residual > maximum_relative_response_residual:
        raise RuntimeError("语义条件子空间的相对响应残差超过正式门禁")
    return result


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

    detection_index = config.injection_step_indices[0]
    payload = (
        f"slm_wm_image_only_attention|{config.carrier_model_reference}|{config.width}x{config.height}|"
        f"{config.inference_steps}|{detection_index}"
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

    pipeline.scheduler.set_timesteps(config.inference_steps, device=pipeline._execution_device)
    detection_index = config.injection_step_indices[0]
    timestep = pipeline.scheduler.timesteps[detection_index]
    public_detection_seed = _public_detection_noise_seed(config)

    def extract(image: Any) -> tuple[tuple[str, Any, tuple[int, ...]], ...]:
        """从任意待检图像提取全部冻结层的公开固定噪声 Q/K 关系。"""

        latent = _encode_image_latent(pipeline, image)
        generator = torch.Generator(device=latent.device.type).manual_seed(public_detection_seed)
        noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
        if hasattr(pipeline.scheduler, "scale_noise"):
            noisy_latent = pipeline.scheduler.scale_noise(latent, timestep, noise)
        else:
            sigma = float(detection_index + 1) / float(config.inference_steps)
            noisy_latent = (1.0 - sigma) * latent + sigma * noise
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
    grid = functional.affine_grid(theta, tensor.shape, align_corners=False)
    aligned = functional.grid_sample(tensor, grid, mode="bilinear", padding_mode="border", align_corners=False)
    output = (aligned[0].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).byte().numpy()
    return Image.fromarray(output, mode="RGB")


def run_semantic_watermark_runtime(
    config: SemanticWatermarkRuntimeConfig,
    runtime_context: SemanticWatermarkRuntimeContext | None = None,
) -> tuple[
    SemanticWatermarkRuntimeResult,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    Any,
    Any,
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
    previous_injection_latent: Any | None = None

    def inject(pipe: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        nonlocal previous_injection_latent
        latent = callback_kwargs.get("latents")
        if latent is None or step_index not in config.injection_step_indices:
            return callback_kwargs
        with torch.enable_grad():
            signals = feature_runtime.branch_signal_maps(latent.float(), previous_injection_latent)
            lf_template = build_low_frequency_template(
                latent,
                config.key_material,
                config.carrier_model_reference,
            )
            tail_template, tail_threshold, retained_fraction = build_tail_robust_template(
                latent,
                config.key_material,
                config.carrier_model_reference,
                config.tail_fraction if config.tail_truncation_enabled else 1.0,
            )
            transformer_forward = _transformer_forward_function(
                pipeline,
                timestep,
                unconditional_prompt,
                unconditional_pooled,
            )
            with DifferentiableAttentionRecorder(
                attention_modules,
                max_tokens=config.max_attention_tokens,
            ) as recorder:
                attention_gradient = (
                    compute_attention_geometry_gradient(
                        latent,
                        transformer_forward,
                        recorder,
                        config.key_material,
                    )
                    if config.attention_geometry_enabled
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
                    stability_values=signals["stability"].mean(dim=0).reshape(-1).cpu().tolist(),
                    saliency_values=signals["saliency"].mean(dim=0).reshape(-1).cpu().tolist(),
                    attention_stability_values=attention_stability.mean(dim=0).reshape(-1).cpu().tolist(),
                )
                branch_fields = {
                    "lf_content": risk_bundle.lf_content,
                    "tail_robust": risk_bundle.tail_robust,
                    "attention_geometry": risk_bundle.attention_geometry,
                }
                active_branch_fields = {
                    branch_name: branch_field
                    for branch_name, branch_field in branch_fields.items()
                    if (
                        (branch_name == "lf_content" and config.lf_enabled)
                        or (branch_name == "tail_robust" and config.tail_robust_enabled)
                        or (branch_name == "attention_geometry" and config.attention_geometry_enabled)
                    )
                }
                preferred_directions = {
                    "lf_content": (lf_template,),
                    "tail_robust": (tail_template,),
                    "attention_geometry": (
                        () if attention_gradient is None else (attention_gradient.gradient,)
                    ),
                }
                if config.null_space_enabled:
                    linearized_latent = latent.float()
                    joint_feature_linearization = build_exact_jvp_linearization(
                        feature_runtime.joint_condition_features,
                        linearized_latent,
                    )
                    subspaces = {
                        branch_name: _solve_branch_subspace(
                            linearized_latent,
                            feature_runtime,
                            config.key_material,
                            branch_name,
                            (
                                _branch_budget(signals["semantic"], branch_field)
                                if config.semantic_routing_enabled
                                else tuple(1.0 for _ in branch_field.budget_values)
                            ),
                            config.candidate_count,
                            config.null_rank,
                            joint_feature_linearization,
                            preferred_directions[branch_name],
                            config.maximum_relative_response_residual,
                        )
                        for branch_name, branch_field in active_branch_fields.items()
                    }
                    # 子空间矩阵已经物化; 立即释放可复用线性化图, 为 Q/K 回溯前向腾出显存。
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
                        config.minimum_projection_energy_retention,
                    )
                    if config.lf_enabled
                    else None
                )
                tail_carrier = (
                    project_canonical_template(
                        "tail_robust",
                        tail_template,
                        subspaces["tail_robust"],
                        config.minimum_projection_energy_retention,
                    )
                    if config.tail_robust_enabled
                    else None
                )
                latent_norm = latent.detach().float().norm()
                lf_update = (
                    lf_carrier.scaled_update(config.lf_relative_strength * float(latent_norm.item()))
                    if lf_carrier is not None
                    else torch.zeros_like(latent)
                )
                tail_update = (
                    tail_carrier.scaled_update(config.tail_relative_strength * float(latent_norm.item()))
                    if tail_carrier is not None
                    else torch.zeros_like(latent)
                )
                if attention_gradient is not None:
                    attention_update = optimize_attention_geometry_update(
                        latent=latent,
                        transformer_forward=transformer_forward,
                        recorder=recorder,
                        key_material=config.key_material,
                        safe_subspace=subspaces["attention_geometry"],
                        update_strength=config.attention_relative_strength * float(latent_norm.item()),
                        precomputed_gradient=attention_gradient,
                    )
                    attention_tensor = attention_update.update
                    attention_record = {
                        "attention_score_before": attention_update.score_before,
                        "attention_score_after": attention_update.score_after,
                        "attention_score_gain": attention_update.score_gain,
                        "attention_applied_update_strength": attention_update.applied_update_strength,
                        "attention_backtracking_step_count": attention_update.backtracking_step_count,
                        "attention_update_digest": attention_update.update_digest,
                    }
                else:
                    attention_tensor = torch.zeros_like(latent)
                    attention_record = {
                        "attention_score_before": None,
                        "attention_score_after": None,
                        "attention_score_gain": None,
                        "attention_applied_update_strength": None,
                        "attention_backtracking_step_count": None,
                        "attention_update_digest": "",
                    }
                combined_update = lf_update + tail_update + attention_tensor
                injected = latent + combined_update.to(dtype=latent.dtype)
        update_records.append(
            {
                "run_id": run_id,
                "prompt_id": config.prompt_id,
                "split": config.split,
                "step_index": int(step_index),
                "timestep": float(timestep.detach().float().item()),
                "latent_digest_before": _tensor_digest(latent),
                "latent_digest_after": _tensor_digest(injected),
                "combined_update_digest": _tensor_digest(combined_update),
                "relative_update_norm": tensor_norm(combined_update) / max(tensor_norm(latent), 1e-12),
                "branch_risk_bundle_digest": risk_bundle.bundle_digest,
                "branch_risk_records": {
                    name: _branch_risk_record(branch_field)
                    for name, branch_field in branch_fields.items()
                },
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
                **attention_record,
                "metadata": {
                    "jvp_mode": jvp_modes[0] if len(jvp_modes) == 1 else "disabled_or_mixed",
                    "jvp_modes": list(jvp_modes),
                    "basis_solver": "weighted_response_svd",
                    "attention_source": "real_qk_projection",
                    "detector_requires_generation_trace": False,
                    "semantic_routing_enabled": config.semantic_routing_enabled,
                    "null_space_enabled": config.null_space_enabled,
                    "lf_enabled": config.lf_enabled,
                    "tail_robust_enabled": config.tail_robust_enabled,
                    "tail_truncation_enabled": config.tail_truncation_enabled,
                    "attention_geometry_enabled": config.attention_geometry_enabled,
                    "supports_paper_claim": False,
                },
            }
        )
        previous_injection_latent = latent.detach().float()
        callback_kwargs["latents"] = injected.detach().to(dtype=latent.dtype)
        return callback_kwargs

    watermarked_image = pipeline(
        generator=watermarked_generator,
        callback_on_step_end=inject,
        callback_on_step_end_tensor_inputs=["latents"],
        **common_kwargs,
    ).images[0]
    paired_quality = compute_image_quality_metrics(clean_image, watermarked_image)

    lf_weight = 0.70 if config.lf_enabled and config.tail_robust_enabled else (1.0 if config.lf_enabled else 0.0)
    tail_weight = 1.0 - lf_weight
    detector_config = ImageOnlyDetectionConfig(
        model_id=config.carrier_model_reference,
        content_threshold=config.content_threshold,
        geometry_score_threshold=config.geometry_score_threshold,
        rescue_margin_low=config.rescue_margin_low,
        lf_weight=lf_weight,
        tail_robust_weight=tail_weight,
        tail_fraction=config.tail_fraction if config.tail_truncation_enabled else 1.0,
    )
    attention_extractor = _image_attention_extractor(
        pipeline,
        config,
        attention_modules,
        unconditional_prompt,
        unconditional_pooled,
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
            "detector_input_access_mode": "image_key_public_model_only",
            "supports_paper_claim": False,
            "paired_quality": paired_quality,
        },
    )
    return result, tuple(update_records), tuple(detections), clean_image, watermarked_image, attacked_images


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
    result, update_records, detections, clean_image, watermarked_image, attacked_images = run_semantic_watermark_runtime(
        config,
        runtime_context=runtime_context,
    )
    run_dir = output_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    clean_image_path = run_dir / "clean_image.png"
    watermarked_image_path = run_dir / "watermarked_image.png"
    clean_image.save(clean_image_path)
    watermarked_image.save(watermarked_image_path)
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
    detection_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in governed_detections), encoding="utf-8")
    resolved_result = SemanticWatermarkRuntimeResult(
        **{
            **result.to_dict(),
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
            manifest_path.relative_to(root_path).as_posix(),
            *attacked_image_paths,
        ),
        config={
            **asdict(config),
            "key_material": build_stable_digest({"key_material": config.key_material}),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="调用 experiments.runners.semantic_watermark_runtime.write_semantic_watermark_runtime_outputs",
        metadata={
            "run_id": result.run_id,
            "protocol_decision": result.run_decision,
            "detector_input_access_mode": "image_key_public_model_only",
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest_path.write_text(_stable_json(manifest), encoding="utf-8")
    return resolved_result
