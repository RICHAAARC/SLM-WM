"""Shallow Diffuse 在 SD3.5 Medium 上的方法忠实适配器.

该模块位于 external baseline 适配层. 同一 clean base latent 先按正文 guidance
运行到 edit timestep, 固定 patch 在该状态注入, clean 与 watermarked 分支再以
guidance=1 完成后段去噪. 最终图像仅保留指定水印通道, 其他通道来自 clean 分支.
检测器从图像仅反演到同一 edit timestep, 再计算 masked patch 距离.
整次运行在 Prompt 循环外生成唯一 patch, 所有 Prompt 与恢复会话复用该固定载体.

项目特定写法:
- SD3.5 Medium 使用 16-channel latent, 因此 mask, patch 和通道融合按可配置通道数构造.
- 两段生成与部分反演直接复用 SD3.5 FlowMatch Euler 的同一完整 schedule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.runtime.image_metrics import measured_image_ssim, measured_score_retention
from experiments.protocol.attacks import attack_config_digest
from experiments.protocol.formal_randomization import (
    build_canonical_sd35_base_latent,
    formal_random_trace_fields,
)
from main.core.digest import build_stable_digest

from external_baseline.primary.sd35_method_faithful_common import (
    DEFAULT_SD35_MODEL_ID,
    DEFAULT_SD35_MODEL_REVISION,
    METHOD_FAITHFUL_ADAPTER_BOUNDARY,
    apply_formal_image_attack,
    canonical_attack_family,
    canonical_attack_name,
    circle_mask,
    derive_threshold,
    emit_adapter_progress,
    file_digest,
    formal_image_attack_config,
    image_to_tensor,
    load_prompt_rows,
    load_sd3_pipeline,
    observation_digest,
    prompt_text,
    row_id,
    safe_file_stem,
    select_prompt_rows,
    split_name,
    validate_model_revision,
    validated_observation_attack_identity,
    write_json,
)
from external_baseline.primary.sd35_method_faithful_units import (
    aggregate_method_faithful_unit_records,
    apply_frozen_threshold,
    build_irreversible_random_material_digest,
    build_method_faithful_unit_context,
    build_method_faithful_unit_spec,
    load_completed_method_faithful_unit,
    repository_relative_method_faithful_path,
    resolve_method_faithful_output_path,
    threshold_independent_observation,
    write_completed_method_faithful_unit,
)

BASELINE_ID = "shallow_diffuse"
DEFAULT_SCORE_NAME = "negative_shallow_diffuse_masked_patch_distance"


def build_watermark_mask(
    latent_shape: tuple[int, int, int, int],
    *,
    mask_shape: str,
    radius: int,
    inner_radius: int,
    channel: int,
    device: str,
) -> Any:
    """构造 Shallow Diffuse 使用的局部 latent mask。"""

    import torch

    batch, channels, height, width = latent_shape
    if height != width:
        raise ValueError("shallow_diffuse_sd35_adapter_requires_square_latent")
    if channel < -1 or channel >= channels:
        raise ValueError(f"w_channel 必须在 [-1, {channels - 1}] 范围内")
    mask = torch.zeros(latent_shape, device=device, dtype=torch.bool)
    shape_name = str(mask_shape).strip().lower()
    if shape_name == "circle":
        base_mask = torch.tensor(circle_mask(width, int(radius)), device=device, dtype=torch.bool)
    elif shape_name == "ring":
        outer = torch.tensor(circle_mask(width, int(radius)), device=device, dtype=torch.bool)
        inner = torch.tensor(circle_mask(width, int(inner_radius)), device=device, dtype=torch.bool)
        base_mask = outer & ~inner
    elif shape_name == "square":
        base_mask = torch.zeros((height, width), device=device, dtype=torch.bool)
        anchor = width // 2
        base_mask[anchor - int(radius) : anchor + int(radius), anchor - int(radius) : anchor + int(radius)] = True
    elif shape_name == "whole":
        base_mask = torch.ones((height, width), device=device, dtype=torch.bool)
    elif shape_name == "outercircle":
        base_mask = ~torch.tensor(circle_mask(width, int(radius)), device=device, dtype=torch.bool)
    else:
        raise ValueError(f"unsupported_shallow_diffuse_mask_shape:{mask_shape}")
    if channel == -1:
        mask[:, :] = base_mask
    else:
        mask[:, channel] = base_mask
    if batch != 1:
        raise ValueError("shallow_diffuse_sd35_adapter_currently_requires_batch_one")
    return mask


def build_watermark_patch(
    latent_shape: tuple[int, int, int, int],
    *,
    pattern: str,
    radius: int,
    generator: Any,
    device: str,
) -> Any:
    """按 Shallow Diffuse 的 seed / complex pattern 语义构造 watermark patch。"""

    import torch

    init = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float32)
    pattern_name = str(pattern).strip().lower()
    if "complex2" in pattern_name:
        patch = torch.fft.fft2(init)
    elif "complex" in pattern_name:
        patch = torch.fft.fftshift(torch.fft.fft2(init), dim=(-1, -2))
    else:
        patch = init
    if "zero" in pattern_name:
        return patch * 0
    if "ring" in pattern_name:
        source = patch.clone().detach()
        latent_size = int(latent_shape[-1])
        for current_radius in range(int(radius), 0, -1):
            ring_mask = torch.tensor(circle_mask(latent_size, current_radius), device=device, dtype=torch.bool)
            for channel in range(int(latent_shape[1])):
                patch[:, channel, ring_mask] = source[0, channel, 0, current_radius].item()
        return patch
    if "rand" in pattern_name or "seed" in pattern_name:
        patch[:] = patch[0]
        return patch
    raise ValueError(f"unsupported_shallow_diffuse_pattern:{pattern}")


def build_fixed_watermark_carrier(
    latent_shape: tuple[int, int, int, int],
    *,
    mask_shape: str,
    radius: int,
    inner_radius: int,
    channel: int,
    pattern: str,
    watermark_seed: int,
    device: str,
) -> tuple[Any, Any, str]:
    """由固定 seed 构造整次运行共享的 Shallow Diffuse mask, patch 和摘要."""

    import torch

    mask = build_watermark_mask(
        latent_shape,
        mask_shape=mask_shape,
        radius=int(radius),
        inner_radius=int(inner_radius),
        channel=int(channel),
        device=device,
    )
    patch_generator = torch.Generator(device=device).manual_seed(int(watermark_seed))
    patch = build_watermark_patch(
        latent_shape,
        pattern=pattern,
        radius=int(radius),
        generator=patch_generator,
        device=device,
    )
    carrier_digest_random = build_irreversible_random_material_digest(patch)
    return mask, patch, carrier_digest_random


def inject_watermark(latents: Any, mask: Any, patch: Any, *, injection: str) -> Any:
    """按照 Shallow Diffuse 的 injection 语义在 latent 中写入 patch。"""

    import torch

    injection_name = str(injection).strip().lower()
    if injection_name == "complex":
        latent_fft = torch.fft.fftshift(torch.fft.fft2(latents.float()), dim=(-1, -2))
        latent_fft[mask] = patch[mask].clone()
        injected = torch.fft.ifft2(torch.fft.ifftshift(latent_fft, dim=(-1, -2))).real
        return injected.to(dtype=latents.dtype)
    if injection_name == "complex2":
        latent_fft = torch.fft.fft2(latents.float())
        latent_fft[mask] = patch[mask].clone()
        injected = torch.fft.ifft2(latent_fft).real
        return injected.to(dtype=latents.dtype)
    if "seed" in injection_name:
        updated = latents.clone()
        updated[mask] = patch.to(dtype=updated.dtype)[mask].clone()
        return updated
    raise ValueError(f"unsupported_shallow_diffuse_injection:{injection}")


def score_latents(reversed_latents: Any, *, mask: Any, patch: Any, measurement: str) -> float:
    """计算 Shallow Diffuse 检测分数, 分数越大表示越像阳性。"""

    import torch

    measurement_name = str(measurement).strip().lower()
    if "complex2" in measurement_name:
        recovered = torch.fft.fft2(reversed_latents.float())
        target = patch
    elif "complex" in measurement_name:
        recovered = torch.fft.fftshift(torch.fft.fft2(reversed_latents.float()), dim=(-1, -2))
        target = patch
    elif "seed" in measurement_name:
        recovered = reversed_latents.float()
        target = patch.float()
    else:
        raise ValueError(f"unsupported_shallow_diffuse_measurement:{measurement}")
    if "l1" not in measurement_name:
        raise ValueError(f"unsupported_shallow_diffuse_measurement:{measurement}")
    distance = torch.abs(recovered[mask] - target[mask]).mean().detach().cpu().item()
    return -float(distance)


def build_observation(
    *,
    event_id: str,
    score: float,
    threshold: float,
    threshold_source: str,
    row: dict[str, Any],
    index: int,
    sample_role: str,
    attack_family: str,
    attack_condition: str,
    image_id: str,
    image_path: str,
    image_digest: str,
    latent_shape: tuple[int, int, int, int],
    execution_device: str,
    model_id: str,
    model_revision: str,
    injection_mode: str,
    quality_score: float,
    score_retention: float,
    attack_id: str = "",
    resource_profile: str = "",
    attack_config_digest_value: str = "",
) -> dict[str, Any]:
    """构造统一 baseline observation。"""

    detection_decision = bool(float(score) >= float(threshold))
    attack_identity = validated_observation_attack_identity(
        sample_role=sample_role,
        attack_family=attack_family,
        attack_name=attack_condition,
        attack_id=attack_id,
        resource_profile=resource_profile,
        attack_config_digest_value=attack_config_digest_value,
    )
    return observation_digest(
        {
            "event_id": event_id,
            "baseline_id": BASELINE_ID,
            "score": float(score),
            "threshold": float(threshold),
            "score_name": DEFAULT_SCORE_NAME,
            "higher_is_positive": True,
            "detection_decision": detection_decision,
            "final_decision": detection_decision,
            "split": split_name(row),
            "sample_role": sample_role,
            "attack_family": attack_family,
            "attack_name": attack_condition,
            "attack_condition": attack_condition,
            **attack_identity,
            "prompt_id": row_id(row, index, "prompt_id", "prompt"),
            "prompt_text": prompt_text(row),
            "randomization_repeat_id": str(row["randomization_repeat_id"]),
            "generation_seed_index": int(row["generation_seed_index"]),
            "generation_seed_offset": int(row["generation_seed_offset"]),
            "generation_seed_random": int(row["generation_seed_random"]),
            "watermark_key_index": int(row["watermark_key_index"]),
            "watermark_key_seed_random": int(
                row["watermark_key_seed_random"]
            ),
            "watermark_key_material_digest_random": str(
                row["watermark_key_material_digest_random"]
            ),
            "formal_randomization_protocol_digest": str(
                row["formal_randomization_protocol_digest"]
            ),
            "formal_randomization_identity_digest_random": str(
                row["formal_randomization_identity_digest_random"]
            ),
            "base_latent_content_digest_random": str(
                row["base_latent_content_digest_random"]
            ),
            "base_latent_identity_digest_random": str(
                row["base_latent_identity_digest_random"]
            ),
            "image_id": image_id,
            "image_path": image_path,
            "image_digest": image_digest,
            "producer_id": "shallow_diffuse_method_faithful_sd35_adapter",
            "producer_role": "external_baseline_method_faithful_adapter",
            "adapter_boundary": METHOD_FAITHFUL_ADAPTER_BOUNDARY,
            "threshold_source": threshold_source,
            "formal_result_claim": False,
            "supports_paper_claim": False,
            "generation_model_id": model_id,
            "generation_model_revision": validate_model_revision(model_revision),
            "latent_shape": list(latent_shape),
            "execution_device": execution_device,
            "shallow_injection_mode": injection_mode,
            "quality_score": float(quality_score),
            "score_retention": float(score_retention),
        }
    )


def resolve_shallow_diffuse_edit_timestep(
    num_inference_steps: int,
    edit_fraction: float,
) -> tuple[int, int]:
    """按官方 floor 语义返回 edit timestep 与前段结束 schedule index."""

    step_count = int(num_inference_steps)
    if step_count <= 1:
        raise ValueError("Shallow Diffuse 至少需要2个推理 step")
    edit_timestep = int(float(edit_fraction) * step_count)
    if not 0 < edit_timestep < step_count:
        raise ValueError("Shallow Diffuse edit timestep 必须位于完整 schedule 内部")
    return edit_timestep, step_count - edit_timestep


def _scheduler_config_value(config: Any, field_name: str, default: Any) -> Any:
    """从 Diffusers FrozenDict 或普通对象读取 scheduler 配置."""

    if hasattr(config, "get"):
        return config.get(field_name, default)
    return getattr(config, field_name, default)


def _set_flow_matching_schedule(
    pipe: Any,
    *,
    num_inference_steps: int,
    latent_shape: tuple[int, ...],
) -> tuple[Any, Any]:
    """按 SD3.5 pipeline 语义设置完整 FlowMatch Euler schedule."""

    scheduler_kwargs: dict[str, Any] = {}
    scheduler_config = pipe.scheduler.config
    if bool(
        _scheduler_config_value(
            scheduler_config,
            "use_dynamic_shifting",
            False,
        )
    ):
        patch_size = int(getattr(pipe.transformer.config, "patch_size", 2))
        image_sequence_length = (int(latent_shape[-2]) // patch_size) * (
            int(latent_shape[-1]) // patch_size
        )
        base_sequence_length = int(
            _scheduler_config_value(
                scheduler_config,
                "base_image_seq_len",
                256,
            )
        )
        maximum_sequence_length = int(
            _scheduler_config_value(
                scheduler_config,
                "max_image_seq_len",
                4096,
            )
        )
        base_shift = float(
            _scheduler_config_value(scheduler_config, "base_shift", 0.5)
        )
        maximum_shift = float(
            _scheduler_config_value(scheduler_config, "max_shift", 1.16)
        )
        slope = (maximum_shift - base_shift) / (
            maximum_sequence_length - base_sequence_length
        )
        scheduler_kwargs["mu"] = (
            image_sequence_length * slope
            + base_shift
            - base_sequence_length * slope
        )
    pipe.scheduler.set_timesteps(
        int(num_inference_steps),
        device=pipe._execution_device,
        **scheduler_kwargs,
    )
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas
    if len(timesteps) != int(num_inference_steps) or len(sigmas) != len(timesteps) + 1:
        raise RuntimeError("Shallow Diffuse FlowMatch schedule 长度不符合 Euler 契约")
    if bool(_scheduler_config_value(scheduler_config, "stochastic_sampling", False)):
        raise RuntimeError("Shallow Diffuse 正式路线要求确定性 FlowMatch Euler schedule")
    return timesteps, sigmas


def _encode_sd3_prompt_conditioning(
    pipe: Any,
    *,
    prompt: str,
    guidance_scale: float,
) -> tuple[Any, Any, bool]:
    """按 SD3 pipeline 语义构造正负 Prompt embedding 与 pooled projection."""

    import torch

    do_classifier_free_guidance = float(guidance_scale) > 1.0
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        device=pipe._execution_device,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt="",
    )
    if do_classifier_free_guidance:
        if negative_prompt_embeds is None or negative_pooled_prompt_embeds is None:
            raise RuntimeError("Shallow Diffuse 前段 CFG 缺少 negative embedding")
        prompt_embeds = torch.cat(
            [negative_prompt_embeds, prompt_embeds],
            dim=0,
        )
        pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds],
            dim=0,
        )
    return prompt_embeds, pooled_prompt_embeds, do_classifier_free_guidance


def _predict_flow_matching_velocity(
    pipe: Any,
    latents: Any,
    *,
    timestep: Any,
    prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    guidance_scale: float,
    do_classifier_free_guidance: bool,
) -> Any:
    """调用真实 SD3 transformer 并执行与 pipeline 一致的 CFG 合成."""

    import torch

    latent_model_input = (
        torch.cat([latents] * 2)
        if do_classifier_free_guidance
        else latents
    )
    timestep_tensor = timestep.expand(latent_model_input.shape[0])
    velocity = pipe.transformer(
        hidden_states=latent_model_input,
        timestep=timestep_tensor,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        joint_attention_kwargs=getattr(pipe, "_joint_attention_kwargs", None),
        return_dict=False,
    )[0]
    if do_classifier_free_guidance:
        velocity_unconditional, velocity_conditional = velocity.chunk(2)
        velocity = velocity_unconditional + float(guidance_scale) * (
            velocity_conditional - velocity_unconditional
        )
    return velocity


def denoise_flow_matching_segment(
    pipe: Any,
    latents: Any,
    *,
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    start_schedule_index: int,
    end_schedule_index: int,
) -> Any:
    """沿完整 SD3 schedule 执行指定的前闭后开真实 Euler 去噪区间."""

    import torch

    step_count = int(num_inference_steps)
    start_index = int(start_schedule_index)
    end_index = int(end_schedule_index)
    if not 0 <= start_index <= end_index <= step_count:
        raise ValueError("Shallow Diffuse 去噪区间超出完整 schedule")
    with torch.inference_mode():
        timesteps, sigmas = _set_flow_matching_schedule(
            pipe,
            num_inference_steps=step_count,
            latent_shape=tuple(latents.shape),
        )
        prompt_embeds, pooled_prompt_embeds, do_cfg = (
            _encode_sd3_prompt_conditioning(
                pipe,
                prompt=prompt,
                guidance_scale=float(guidance_scale),
            )
        )
        current = latents.clone()
        for schedule_index in range(start_index, end_index):
            velocity = _predict_flow_matching_velocity(
                pipe,
                current,
                timestep=timesteps[schedule_index],
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=float(guidance_scale),
                do_classifier_free_guidance=do_cfg,
            )
            sigma_current = sigmas[schedule_index].to(
                device=current.device,
                dtype=torch.float32,
            )
            sigma_next = sigmas[schedule_index + 1].to(
                device=current.device,
                dtype=torch.float32,
            )
            current = (
                current.to(dtype=torch.float32)
                + (sigma_next - sigma_current) * velocity
            ).to(dtype=velocity.dtype)
    return current


def fuse_shallow_diffuse_watermark_channels(
    clean_latents: Any,
    watermarked_latents: Any,
    *,
    watermark_channel: int,
) -> Any:
    """仅保留指定水印通道, 其余通道精确恢复 clean 分支."""

    if tuple(clean_latents.shape) != tuple(watermarked_latents.shape):
        raise ValueError("Shallow Diffuse 通道融合要求相同 latent 形状")
    channel = int(watermark_channel)
    if channel < -1 or channel >= int(clean_latents.shape[1]):
        raise ValueError("Shallow Diffuse 水印通道超出 latent 维度")
    fused = clean_latents.clone()
    if channel == -1:
        fused[:] = watermarked_latents
    else:
        fused[:, channel, :, :] = watermarked_latents[:, channel, :, :]
    return fused


def generate_shallow_diffuse_latent_pair(
    pipe: Any,
    prompt: str,
    *,
    base_latents: Any,
    mask: Any,
    patch: Any,
    num_inference_steps: int,
    guidance_scale: float,
    edit_fraction: float,
    injection: str,
    watermark_channel: int,
) -> tuple[Any, Any, dict[str, Any]]:
    """执行官方两段去噪, edit 注入, guidance=1 后段和通道融合."""

    edit_timestep, edit_schedule_index = resolve_shallow_diffuse_edit_timestep(
        int(num_inference_steps),
        float(edit_fraction),
    )
    edit_clean_latents = denoise_flow_matching_segment(
        pipe,
        base_latents,
        prompt=prompt,
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        start_schedule_index=0,
        end_schedule_index=edit_schedule_index,
    )
    edit_watermarked_latents = inject_watermark(
        edit_clean_latents.clone(),
        mask,
        patch,
        injection=str(injection),
    )
    clean_final_latents = denoise_flow_matching_segment(
        pipe,
        edit_clean_latents,
        prompt=prompt,
        guidance_scale=1.0,
        num_inference_steps=int(num_inference_steps),
        start_schedule_index=edit_schedule_index,
        end_schedule_index=int(num_inference_steps),
    )
    watermarked_branch_latents = denoise_flow_matching_segment(
        pipe,
        edit_watermarked_latents,
        prompt=prompt,
        guidance_scale=1.0,
        num_inference_steps=int(num_inference_steps),
        start_schedule_index=edit_schedule_index,
        end_schedule_index=int(num_inference_steps),
    )
    fused_watermarked_latents = fuse_shallow_diffuse_watermark_channels(
        clean_final_latents,
        watermarked_branch_latents,
        watermark_channel=int(watermark_channel),
    )
    protocol = {
        "injection_mode": "edit_timestep_split_flow_matching",
        "edit_timestep": edit_timestep,
        "edit_schedule_index": edit_schedule_index,
        "pre_edit_guidance_scale": float(guidance_scale),
        "post_edit_guidance_scale": 1.0,
        "watermark_channel": int(watermark_channel),
        "channel_fusion": "watermark_channel_from_watermarked_branch_other_channels_from_clean_branch",
    }
    return clean_final_latents, fused_watermarked_latents, protocol


def decode_sd3_latents_to_image(pipe: Any, latents: Any) -> Any:
    """按 StableDiffusion3Pipeline 当前解码契约把 latent 转换为 PIL 图像."""

    import torch

    with torch.inference_mode():
        scaling_factor = float(pipe.vae.config.scaling_factor)
        shift_factor = float(pipe.vae.config.shift_factor)
        decoded = pipe.vae.decode(
            latents / scaling_factor + shift_factor,
            return_dict=False,
        )[0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def invert_flow_matching_to_edit_timestep(
    pipe: Any,
    image_latents: Any,
    *,
    num_inference_steps: int,
    edit_timestep: int,
) -> Any:
    """从图像 latent 仅反演到生成时的同一 edit timestep."""

    import torch

    step_count = int(num_inference_steps)
    resolved_edit_timestep = int(edit_timestep)
    if not 0 < resolved_edit_timestep < step_count:
        raise ValueError("Shallow Diffuse 检测 edit timestep 超出完整 schedule")
    edit_schedule_index = step_count - resolved_edit_timestep
    with torch.inference_mode():
        timesteps, sigmas = _set_flow_matching_schedule(
            pipe,
            num_inference_steps=step_count,
            latent_shape=tuple(image_latents.shape),
        )
        prompt_embeds, pooled_prompt_embeds, do_cfg = (
            _encode_sd3_prompt_conditioning(
                pipe,
                prompt="",
                guidance_scale=1.0,
            )
        )
        current = image_latents.clone()
        for schedule_index in range(step_count - 1, edit_schedule_index - 1, -1):
            velocity = _predict_flow_matching_velocity(
                pipe,
                current,
                timestep=timesteps[schedule_index],
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=1.0,
                do_classifier_free_guidance=do_cfg,
            )
            sigma_current = sigmas[schedule_index + 1].to(
                device=current.device,
                dtype=torch.float32,
            )
            sigma_next = sigmas[schedule_index].to(
                device=current.device,
                dtype=torch.float32,
            )
            current = (
                current.to(dtype=torch.float32)
                + (sigma_next - sigma_current) * velocity
            ).to(dtype=velocity.dtype)
    return current


def score_image(
    pipe: Any,
    image: Any,
    *,
    size: int,
    device: str,
    mask: Any,
    patch: Any,
    measurement: str,
    num_inference_steps: int,
    edit_timestep: int,
) -> float:
    """把图像仅反演到 edit timestep 并计算 masked patch 距离分数."""

    tensor = image_to_tensor(
        image,
        size=int(size),
        device=device,
        dtype=pipe.vae.dtype,
    )
    image_latents = pipe.get_image_latents(tensor, sample=False)
    reversed_latents = invert_flow_matching_to_edit_timestep(
        pipe,
        image_latents,
        num_inference_steps=int(num_inference_steps),
        edit_timestep=int(edit_timestep),
    )
    return score_latents(
        reversed_latents,
        mask=mask,
        patch=patch,
        measurement=measurement,
    )


def run_shallow_diffuse_method_faithful_adapter(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行 Shallow Diffuse SD3.5 方法忠实适配流程。"""

    import torch
    from PIL import Image

    prompt_rows = select_prompt_rows(load_prompt_rows(args.prompt_plan), args.max_samples)
    attack_families = [item.strip() for item in str(args.attack_families or "").split(",") if item.strip()]
    if int(args.num_inversion_steps) != int(args.num_inference_steps):
        raise ValueError(
            "Shallow Diffuse 检测必须复用生成的完整 FlowMatch schedule"
        )
    edit_timestep, edit_schedule_index = resolve_shallow_diffuse_edit_timestep(
        int(args.num_inference_steps),
        float(args.edit_fraction),
    )
    test_prompt_count = sum(split_name(row) == "test" for row in prompt_rows)
    progress_total = max(1, 1 + len(prompt_rows) + len(attack_families) * test_prompt_count * 2)
    progress_completed = 0
    emit_adapter_progress(
        baseline_id=BASELINE_ID,
        operation="load_sd3_pipeline",
        completed=progress_completed,
        total=progress_total,
        profile=f"operation=load_sd3_pipeline prompts={len(prompt_rows)} attacks={len(attack_families)}",
    )
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    if args.require_cuda and device != "cuda":
        raise RuntimeError("shallow_diffuse_method_faithful_sd35_requires_cuda")

    pipe = load_sd3_pipeline(
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=device,
        torch_dtype_name=args.torch_dtype,
        adapter_class_name="ShallowDiffuseInversionStableDiffusion3Pipeline",
    )
    progress_completed += 1
    emit_adapter_progress(
        baseline_id=BASELINE_ID,
        operation="source_pair_generation",
        completed=progress_completed,
        total=progress_total,
        profile=f"operation=source_pair_generation prompt=0/{len(prompt_rows)}",
    )
    artifact_root = Path(args.artifact_root) if args.artifact_root else Path(args.out).resolve().parent / "artifacts"
    clean_dir = artifact_root / "images" / "clean"
    watermarked_dir = artifact_root / "images" / "watermarked"
    attacked_dir = artifact_root / "images" / "attacked"
    clean_dir.mkdir(parents=True, exist_ok=True)
    watermarked_dir.mkdir(parents=True, exist_ok=True)
    attacked_dir.mkdir(parents=True, exist_ok=True)

    latent_shape = (1, int(args.latent_channels), max(1, int(args.height) // 8), max(1, int(args.width) // 8))
    mask, patch, watermark_carrier_digest_random = build_fixed_watermark_carrier(
        latent_shape,
        mask_shape=str(args.w_mask_shape),
        radius=int(args.w_radius),
        inner_radius=int(args.w_inner_radius),
        channel=int(args.w_channel),
        pattern=str(args.w_pattern),
        watermark_seed=int(args.watermark_seed),
        device=device,
    )
    run_config = {
        "adapter_boundary": METHOD_FAITHFUL_ADAPTER_BOUNDARY,
        "model_id": args.model_id,
        "model_revision": validate_model_revision(args.model_revision),
        "torch_dtype": str(args.torch_dtype),
        "latent_shape": list(latent_shape),
        "height": int(args.height),
        "width": int(args.width),
        "num_inference_steps": int(args.num_inference_steps),
        "num_inversion_steps": int(args.num_inversion_steps),
        "guidance_scale": float(args.guidance_scale),
        "target_fpr": float(args.target_fpr),
        "explicit_threshold": args.threshold,
        "attack_families": list(attack_families),
        "attack_execution_split": "test",
        "prompt_count": len(prompt_rows),
        "test_prompt_count": test_prompt_count,
        "seed": int(args.seed),
        "watermark_seed": int(args.watermark_seed),
        "w_channel": int(args.w_channel),
        "w_radius": int(args.w_radius),
        "w_inner_radius": int(args.w_inner_radius),
        "w_mask_shape": str(args.w_mask_shape),
        "w_pattern": str(args.w_pattern),
        "w_injection": str(args.w_injection),
        "w_measurement": str(args.w_measurement),
        "edit_fraction": float(args.edit_fraction),
        "edit_timestep": edit_timestep,
        "edit_schedule_index": edit_schedule_index,
        "pre_edit_guidance_scale": float(args.guidance_scale),
        "post_edit_guidance_scale": 1.0,
        "detection_inversion_stop_timestep": edit_timestep,
        "channel_fusion": "watermark_channel_from_watermarked_branch_other_channels_from_clean_branch",
        "watermark_carrier_digest_random": watermark_carrier_digest_random,
        "prompt_plan_digest": build_stable_digest(prompt_rows),
    }
    unit_context = build_method_faithful_unit_context(
        baseline_id=BASELINE_ID,
        artifact_root=artifact_root,
        run_config=run_config,
        execution_device=device,
        torch_module=torch,
    )
    observations_without_threshold: list[dict[str, Any]] = []
    image_pairs: list[dict[str, Any]] = []
    runtime_keys: dict[str, dict[str, Any]] = {}
    injection_modes: set[str] = set()
    source_unit_specs: list[Any] = []
    source_unit_records: list[dict[str, Any]] = []

    for index, row in enumerate(prompt_rows, start=1):
        current_prompt = prompt_text(row)
        prompt_id = row_id(row, index, "prompt_id", "prompt")
        image_id = row_id(row, index, "image_id", "shallow_diffuse_image")
        file_stem = safe_file_stem(image_id, f"shallow_diffuse_image_{index:05d}")
        generation_seed_random = int(row["generation_seed_random"])
        if generation_seed_random != int(args.seed) + index - 1:
            raise RuntimeError("Shallow Diffuse Prompt 生成种子未匹配正式随机化计划")
        if int(row["watermark_key_seed_random"]) != int(args.watermark_seed):
            raise RuntimeError("Shallow Diffuse 水印密钥未匹配正式随机化计划")
        base_latents, base_latent_identity = (
            build_canonical_sd35_base_latent(
                shape=latent_shape,
                generation_seed_random=generation_seed_random,
                model_id=args.model_id,
                model_revision=args.model_revision,
                device=device,
                dtype=pipe.transformer.dtype,
            )
        )
        row = {**row, **base_latent_identity}
        base_latent_content_digest_random = str(
            base_latent_identity["base_latent_content_digest_random"]
        )
        source_unit_spec = build_method_faithful_unit_spec(
            unit_context,
            unit_kind="source_pair",
            row=row,
            index=index,
            random_identity_random={
                "generation_seed_random": generation_seed_random,
                "watermark_seed_random": int(args.watermark_seed),
                "watermark_carrier_digest_random": watermark_carrier_digest_random,
                "base_latent_content_digest_random": base_latent_content_digest_random,
                **formal_random_trace_fields(base_latent_identity),
            },
            unit_parameters={
                "image_id": image_id,
                "latent_shape": list(latent_shape),
                "edit_timestep": edit_timestep,
                "edit_schedule_index": edit_schedule_index,
                "post_edit_guidance_scale": 1.0,
                "watermark_channel": int(args.w_channel),
            },
        )
        source_unit_specs.append(source_unit_spec)
        completed_source_unit = load_completed_method_faithful_unit(
            unit_context,
            source_unit_spec,
        )
        if completed_source_unit is not None:
            unit_data = completed_source_unit["unit_data"]
            image_pair = dict(unit_data["image_pair"])
            source_observations = [
                dict(item) for item in unit_data["observations_without_threshold"]
            ]
            if len(source_observations) != 2:
                raise ValueError("Shallow Diffuse Prompt 源图完成单元必须包含两条连续分数")
            injection_mode = str(image_pair["shallow_injection_mode"])
            injection_modes.add(injection_mode)
            image_pairs.append(image_pair)
            observations_without_threshold.extend(source_observations)
            runtime_keys[image_id] = {
                "mask": mask,
                "patch": patch,
                "row": row,
                "row_index": index,
                "injection_mode": injection_mode,
                "clean_score": float(source_observations[0]["score"]),
                "watermarked_score": float(source_observations[1]["score"]),
                "base_latent_content_digest_random": base_latent_content_digest_random,
            }
            source_unit_records.append(completed_source_unit)
            progress_completed += 1
            emit_adapter_progress(
                baseline_id=BASELINE_ID,
                operation="source_pair_resume",
                completed=progress_completed,
                total=progress_total,
                profile=f"operation=source_pair_resume prompt={index}/{len(prompt_rows)} image_id={image_id}",
            )
            continue

        clean_final_latents, watermarked_final_latents, shallow_protocol = (
            generate_shallow_diffuse_latent_pair(
                pipe,
                current_prompt,
                base_latents=base_latents,
                mask=mask,
                patch=patch,
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=float(args.guidance_scale),
                edit_fraction=float(args.edit_fraction),
                injection=str(args.w_injection),
                watermark_channel=int(args.w_channel),
            )
        )
        clean_image = decode_sd3_latents_to_image(pipe, clean_final_latents)
        watermarked_image = decode_sd3_latents_to_image(
            pipe,
            watermarked_final_latents,
        )
        injection_mode = str(shallow_protocol["injection_mode"])
        injection_modes.add(injection_mode)

        clean_path = clean_dir / f"{file_stem}_clean.png"
        watermarked_path = watermarked_dir / f"{file_stem}_shallow_diffuse.png"
        clean_image.save(clean_path)
        watermarked_image.save(watermarked_path)
        clean_digest = file_digest(clean_path)
        watermarked_digest = file_digest(watermarked_path)
        clean_score = score_image(
            pipe,
            clean_image,
            size=int(args.height),
            device=device,
            mask=mask,
            patch=patch,
            measurement=str(args.w_measurement),
            num_inference_steps=int(args.num_inference_steps),
            edit_timestep=edit_timestep,
        )
        watermarked_score = score_image(
            pipe,
            watermarked_image,
            size=int(args.height),
            device=device,
            mask=mask,
            patch=patch,
            measurement=str(args.w_measurement),
            num_inference_steps=int(args.num_inference_steps),
            edit_timestep=edit_timestep,
        )
        pair_quality = measured_image_ssim(clean_image, watermarked_image)

        runtime_keys[image_id] = {
            "mask": mask,
            "patch": patch,
            "row": row,
            "row_index": index,
            "injection_mode": injection_mode,
            "clean_score": clean_score,
            "watermarked_score": watermarked_score,
            "base_latent_content_digest_random": base_latent_content_digest_random,
        }
        image_pairs.append(
            {
                "event_id": image_id,
                "image_id": image_id,
                "prompt_id": prompt_id,
                "prompt_text": current_prompt,
                "split": split_name(row),
                "clean_image_path": str(clean_path),
                "clean_image_digest": clean_digest,
                "watermarked_image_path": str(watermarked_path),
                "watermarked_image_digest": watermarked_digest,
                "baseline_id": BASELINE_ID,
                "generation_model_id": args.model_id,
                "generation_model_revision": args.model_revision,
                "latent_shape": list(latent_shape),
                "shallow_injection_mode": injection_mode,
                "randomization_repeat_id": str(
                    row["randomization_repeat_id"]
                ),
                "generation_seed_index": int(row["generation_seed_index"]),
                "generation_seed_offset": int(row["generation_seed_offset"]),
                "watermark_key_index": int(row["watermark_key_index"]),
                "watermark_key_seed_random": int(
                    row["watermark_key_seed_random"]
                ),
                "watermark_key_material_digest_random": str(
                    row["watermark_key_material_digest_random"]
                ),
                "formal_randomization_identity_digest_random": str(
                    row["formal_randomization_identity_digest_random"]
                ),
                **base_latent_identity,
                "edit_timestep": edit_timestep,
                "edit_schedule_index": edit_schedule_index,
                "post_edit_guidance_scale": 1.0,
                "watermark_channel": int(args.w_channel),
                "channel_fusion": shallow_protocol["channel_fusion"],
            }
        )
        observations_without_threshold.append(
            build_observation(
                event_id=f"{image_id}__clean_negative",
                score=clean_score,
                threshold=0.0,
                threshold_source="pending",
                row=row,
                index=index,
                sample_role="clean_negative",
                attack_family="clean",
                attack_condition="clean_none",
                image_id=image_id,
                image_path=str(clean_path),
                image_digest=clean_digest,
                latent_shape=latent_shape,
                execution_device=device,
                model_id=args.model_id,
                model_revision=args.model_revision,
                injection_mode=injection_mode,
                quality_score=1.0,
                score_retention=1.0,
            )
        )
        observations_without_threshold.append(
            build_observation(
                event_id=f"{image_id}__positive_source",
                score=watermarked_score,
                threshold=0.0,
                threshold_source="pending",
                row=row,
                index=index,
                sample_role="positive_source",
                attack_family="clean",
                attack_condition="clean_none",
                image_id=image_id,
                image_path=str(watermarked_path),
                image_digest=watermarked_digest,
                latent_shape=latent_shape,
                execution_device=device,
                model_id=args.model_id,
                model_revision=args.model_revision,
                injection_mode=injection_mode,
                quality_score=pair_quality,
                score_retention=1.0,
            )
        )
        source_observations = [
            threshold_independent_observation(item)
            for item in observations_without_threshold[-2:]
        ]
        observations_without_threshold[-2:] = source_observations
        completed_source_unit = write_completed_method_faithful_unit(
            unit_context,
            source_unit_spec,
            unit_data={
                "image_pair": image_pairs[-1],
                "observations_without_threshold": source_observations,
            },
            artifact_paths=(clean_path, watermarked_path),
        )
        canonical_source_data = completed_source_unit["unit_data"]
        image_pairs[-1] = dict(canonical_source_data["image_pair"])
        observations_without_threshold[-2:] = [
            dict(item)
            for item in canonical_source_data["observations_without_threshold"]
        ]
        source_unit_records.append(completed_source_unit)
        if device == "cuda":
            torch.cuda.empty_cache()
        progress_completed += 1
        emit_adapter_progress(
            baseline_id=BASELINE_ID,
            operation="source_pair_generation",
            completed=progress_completed,
            total=progress_total,
            profile=f"operation=source_pair_generation prompt={index}/{len(prompt_rows)} image_id={image_id}",
        )

    threshold, threshold_source = derive_threshold(observations_without_threshold, args.threshold, args.target_fpr)
    attacked_records: list[dict[str, Any]] = []
    attack_observations_without_threshold: list[dict[str, Any]] = []
    attack_unit_specs: list[Any] = []
    attack_unit_records: list[dict[str, Any]] = []
    for attack_family in attack_families:
        formal_attack_config = formal_image_attack_config(attack_family)
        attack_matrix_family = canonical_attack_family(attack_family)
        attack_matrix_name = canonical_attack_name(attack_family)
        formal_attack_digest = attack_config_digest(formal_attack_config)
        for pair_index, pair in enumerate(image_pairs, start=1):
            if str(pair.get("split", "")) != "test":
                continue
            image_id = str(pair["image_id"])
            runtime = runtime_keys[image_id]
            for role_name, source_path_field, source_digest_field, sample_role in (
                ("clean", "clean_image_path", "clean_image_digest", "attacked_negative"),
                ("watermarked", "watermarked_image_path", "watermarked_image_digest", "attacked_positive"),
            ):
                attack_seed = int(args.seed) + pair_index
                attack_unit_spec = build_method_faithful_unit_spec(
                    unit_context,
                    unit_kind=f"formal_attack_{attack_matrix_name}_{role_name}",
                    row=runtime["row"],
                    index=int(runtime["row_index"]),
                    random_identity_random={
                        "generation_seed_random": int(args.seed) + pair_index - 1,
                        "watermark_seed_random": int(args.watermark_seed),
                        "watermark_carrier_digest_random": watermark_carrier_digest_random,
                        "base_latent_content_digest_random": runtime[
                            "base_latent_content_digest_random"
                        ],
                        "attack_seed_random": attack_seed,
                    },
                    unit_parameters={
                        "image_id": image_id,
                        "source_role": role_name,
                        "sample_role": sample_role,
                        "attack_id": formal_attack_config.attack_id,
                        "attack_config_digest": formal_attack_digest,
                        "edit_timestep": edit_timestep,
                        "edit_schedule_index": edit_schedule_index,
                        "frozen_threshold_digest": build_stable_digest(
                            {
                                "threshold": float(threshold),
                                "threshold_source": threshold_source,
                            }
                        ),
                    },
                )
                attack_unit_specs.append(attack_unit_spec)
                completed_attack_unit = load_completed_method_faithful_unit(
                    unit_context,
                    attack_unit_spec,
                )
                if completed_attack_unit is not None:
                    unit_data = completed_attack_unit["unit_data"]
                    attack_observations_without_threshold.append(
                        dict(unit_data["observation_without_threshold"])
                    )
                    attacked_records.append(dict(unit_data["attacked_record"]))
                    attack_unit_records.append(completed_attack_unit)
                    progress_completed += 1
                    emit_adapter_progress(
                        baseline_id=BASELINE_ID,
                        operation="formal_image_attack_resume",
                        completed=progress_completed,
                        total=progress_total,
                        profile=(
                            "operation=formal_image_attack_resume "
                            f"attack={attack_matrix_name} pair={pair_index}/{len(image_pairs)} role={role_name}"
                        ),
                        attack_name=attack_matrix_name,
                        image_id=image_id,
                    )
                    continue
                source_image_path = resolve_method_faithful_output_path(
                    unit_context,
                    pair[source_path_field],
                )
                with Image.open(source_image_path) as source_image:
                    attacked_image, attack_transform_name, attack_execution = apply_formal_image_attack(
                        source_image,
                        attack_family=attack_family,
                        seed=attack_seed,
                        pipe=pipe,
                        prompt=str(pair["prompt_text"]),
                        size=int(args.height),
                        device=device,
                        detection_score=lambda candidate: score_image(
                            pipe,
                            candidate,
                            size=int(args.height),
                            device=device,
                            mask=runtime["mask"],
                            patch=runtime["patch"],
                            measurement=str(args.w_measurement),
                            num_inference_steps=int(args.num_inference_steps),
                            edit_timestep=edit_timestep,
                        ),
                    )
                    attack_quality = measured_image_ssim(source_image, attacked_image)
                attacked_stem = safe_file_stem(f"{image_id}_{role_name}_{attack_matrix_name}", f"attacked_{pair_index:05d}")
                attacked_path = attacked_dir / f"{attacked_stem}.png"
                attacked_image.save(attacked_path)
                attacked_digest = file_digest(attacked_path)
                attacked_id = f"{image_id}__{role_name}__{attack_matrix_name}"
                score = score_image(
                    pipe,
                    attacked_image,
                    size=int(args.height),
                    device=device,
                    mask=runtime["mask"],
                    patch=runtime["patch"],
                    measurement=str(args.w_measurement),
                    num_inference_steps=int(args.num_inference_steps),
                    edit_timestep=edit_timestep,
                )
                attack_observation = threshold_independent_observation(
                    build_observation(
                        event_id=attacked_id,
                        score=score,
                        threshold=0.0,
                        threshold_source="pending",
                        row=runtime["row"],
                        index=int(runtime["row_index"]),
                        sample_role=sample_role,
                        attack_family=attack_matrix_family,
                        attack_condition=attack_matrix_name,
                        image_id=image_id,
                        image_path=str(attacked_path),
                        image_digest=attacked_digest,
                        latent_shape=latent_shape,
                        execution_device=device,
                        model_id=args.model_id,
                        model_revision=args.model_revision,
                        injection_mode=str(runtime["injection_mode"]),
                        quality_score=attack_quality,
                        score_retention=measured_score_retention(
                            float(runtime[f"{role_name}_score"]),
                            score,
                        ),
                        attack_id=formal_attack_config.attack_id,
                        resource_profile=formal_attack_config.resource_profile,
                        attack_config_digest_value=formal_attack_digest,
                    )
                )
                attacked_record = {
                    "attacked_image_id": attacked_id,
                    "source_image_id": image_id,
                    "source_role": role_name,
                    "sample_role": sample_role,
                    "source_image_path": pair[source_path_field],
                    "source_image_digest": pair[source_digest_field],
                    "attacked_image_path": str(attacked_path),
                    "attacked_image_digest": attacked_digest,
                    "attack_family": attack_matrix_family,
                    "attack_name": attack_matrix_name,
                    "attack_condition": attack_matrix_name,
                    "attack_id": formal_attack_config.attack_id,
                    "resource_profile": formal_attack_config.resource_profile,
                    "attack_config_digest": formal_attack_digest,
                    "attack_transform_name": attack_transform_name,
                    "attack_execution": attack_execution,
                    "generation_model_id": args.model_id,
                    "generation_model_revision": args.model_revision,
                }
                attack_observations_without_threshold.append(attack_observation)
                attacked_records.append(attacked_record)
                completed_attack_unit = write_completed_method_faithful_unit(
                    unit_context,
                    attack_unit_spec,
                    unit_data={
                        "observation_without_threshold": attack_observation,
                        "attacked_record": attacked_record,
                    },
                    artifact_paths=(attacked_path,),
                )
                canonical_attack_data = completed_attack_unit["unit_data"]
                attack_observations_without_threshold[-1] = dict(
                    canonical_attack_data["observation_without_threshold"]
                )
                attacked_records[-1] = dict(canonical_attack_data["attacked_record"])
                attack_unit_records.append(completed_attack_unit)
                if device == "cuda":
                    torch.cuda.empty_cache()
                progress_completed += 1
                emit_adapter_progress(
                    baseline_id=BASELINE_ID,
                    operation="formal_image_attack",
                    completed=progress_completed,
                    total=progress_total,
                    profile=(
                        "operation=formal_image_attack "
                        f"attack={attack_matrix_name} pair={pair_index}/{len(image_pairs)} role={role_name}"
                    ),
                    attack_name=attack_matrix_name,
                    image_id=image_id,
                )

    unit_aggregate = aggregate_method_faithful_unit_records(
        unit_context,
        (*source_unit_records, *attack_unit_records),
        expected_specs=(*source_unit_specs, *attack_unit_specs),
    )
    observations = apply_frozen_threshold(
        (*observations_without_threshold, *attack_observations_without_threshold),
        threshold=threshold,
        threshold_source=threshold_source,
    )
    image_pairs_path = write_json(artifact_root / "shallow_diffuse_image_pairs.json", image_pairs)
    attacked_manifest_path = write_json(
        artifact_root / "attacked_image_manifest.json",
        {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "attacked_images": attacked_records,
            "attacked_image_count": len(attacked_records),
        },
    )
    manifest = {
        "artifact_name": "shallow_diffuse_method_faithful_sd35_adapter_manifest.json",
        "producer_id": "shallow_diffuse_method_faithful_sd35_adapter",
        "baseline_id": BASELINE_ID,
        "adapter_boundary": METHOD_FAITHFUL_ADAPTER_BOUNDARY,
        "adapter_status": "method_faithful_sd35_adapter_ready",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "prompt_plan_path": repository_relative_method_faithful_path(
            unit_context, args.prompt_plan
        ),
        "baseline_observations_path": repository_relative_method_faithful_path(
            unit_context, args.out
        ),
        "artifact_root": repository_relative_method_faithful_path(
            unit_context, artifact_root
        ),
        "image_pairs_path": repository_relative_method_faithful_path(
            unit_context, image_pairs_path
        ),
        "attacked_image_manifest_path": repository_relative_method_faithful_path(
            unit_context, attacked_manifest_path
        ),
        "image_pair_count": len(image_pairs),
        "attacked_image_count": len(attacked_records),
        "test_prompt_count": test_prompt_count,
        "expected_formal_attack_unit_count": test_prompt_count * len(attack_families) * 2,
        "observation_count": len(observations),
        "latent_shape": list(latent_shape),
        "execution_device": device,
        "generation_protocol": {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "post_edit_guidance_scale": 1.0,
            "edit_timestep": edit_timestep,
            "edit_schedule_index": edit_schedule_index,
            "height": int(args.height),
            "width": int(args.width),
        },
        "detection_protocol": {
            "input_access_mode": "image_only",
            "num_inversion_steps": int(args.num_inversion_steps),
            "detection_inversion_stop_timestep": edit_timestep,
            "detection_inversion_stop_schedule_index": edit_schedule_index,
            "target_fpr": float(args.target_fpr),
        },
        "shallow_injection_modes": sorted(injection_modes),
        "watermark_parameters": {
            "w_channel": int(args.w_channel),
            "w_radius": int(args.w_radius),
            "w_inner_radius": int(args.w_inner_radius),
            "w_mask_shape": str(args.w_mask_shape),
            "w_pattern": str(args.w_pattern),
            "w_injection": str(args.w_injection),
            "w_measurement": str(args.w_measurement),
            "edit_fraction": float(args.edit_fraction),
            "edit_timestep": edit_timestep,
            "edit_schedule_index": edit_schedule_index,
            "post_edit_guidance_scale": 1.0,
            "channel_fusion": "watermark_channel_from_watermarked_branch_other_channels_from_clean_branch",
            "watermark_seed": int(args.watermark_seed),
            "watermark_carrier_digest_random": watermark_carrier_digest_random,
        },
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "run_config": unit_context.run_config,
        "run_config_digest": unit_context.run_config_digest,
        "stable_scientific_execution_identity": unit_context.stable_execution_identity,
        "stable_scientific_execution_identity_digest": unit_context.stable_execution_identity_digest,
        "method_faithful_source_identity": unit_context.source_identity,
        "method_faithful_source_identity_digest": unit_context.source_identity[
            "method_faithful_source_identity_digest"
        ],
        **unit_aggregate,
        "formal_result_claim": False,
        "supports_paper_claim": False,
    }
    manifest["adapter_digest"] = build_stable_digest(manifest)
    emit_adapter_progress(
        baseline_id=BASELINE_ID,
        operation="write_outputs",
        completed=progress_total,
        total=progress_total,
        profile=f"operation=write_outputs observations={len(observations)} attacked_images={len(attacked_records)}",
    )
    return observations, manifest


def build_parser() -> argparse.ArgumentParser:
    """构造 Shallow Diffuse 方法忠实适配器参数解析器。"""

    parser = argparse.ArgumentParser(description="运行 Shallow Diffuse SD3.5 方法忠实 external baseline adapter")
    parser.add_argument("--prompt-plan", required=True, help="共同 prompt 计划 JSON 路径")
    parser.add_argument("--out", required=True, help="baseline observations JSON 输出路径")
    parser.add_argument("--artifact-root", default=None, help="图像、攻击结果和 manifest 输出目录")
    parser.add_argument("--model-id", default=DEFAULT_SD35_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_SD35_MODEL_REVISION)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--num-inversion-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watermark-seed", type=int, default=42)
    parser.add_argument("--w-channel", type=int, default=0)
    parser.add_argument("--w-radius", type=int, default=10)
    parser.add_argument("--w-inner-radius", type=int, default=0)
    parser.add_argument("--w-mask-shape", default="circle", choices=("circle", "ring", "square", "whole", "outercircle"))
    parser.add_argument("--w-pattern", default="complex_rand")
    parser.add_argument("--w-injection", default="complex")
    parser.add_argument("--w-measurement", default="l1_complex")
    parser.add_argument("--edit-fraction", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target-fpr", type=float, required=True)
    parser.add_argument("--attack-families", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def run_cli(argv: list[str] | None = None) -> None:
    """命令行入口, 写出 observations 与 manifest。"""

    args = build_parser().parse_args(argv)
    observations, manifest = run_shallow_diffuse_method_faithful_adapter(args)
    write_json(args.out, observations)
    manifest_path = Path(args.out).with_name("shallow_diffuse_method_faithful_sd35_adapter_manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    run_cli()
