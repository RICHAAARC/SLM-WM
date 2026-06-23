"""Shallow Diffuse 在 SD3.5 Medium 上的方法忠实适配器。

该模块位于 external baseline 适配层。它保留 Shallow Diffuse 的核心机制:
在扩散采样的浅层 latent 中按照局部 mask 写入 watermark patch, 生成图像后通过
图像编码和近似反演恢复 latent, 再以 masked patch 距离作为检测分数。

项目特定写法:
- SD3.5 Medium 使用 16-channel latent, 因此 mask 和 patch 均按可配置通道数构造。
- 优先使用 Diffusers 的 callback_on_step_end 在中间 denoising 位置注入; 若运行环境
  缺少该 callback 能力, 会显式记录 fallback, 但仍不直接声明论文主张。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from main.core.digest import build_stable_digest

from external_baseline.primary.sd35_method_faithful_common import (
    METHOD_FAITHFUL_ADAPTER_BOUNDARY,
    apply_image_attack,
    canonical_attack_family,
    canonical_attack_name,
    circle_mask,
    derive_threshold,
    file_digest,
    load_prompt_rows,
    load_sd3_pipeline,
    observation_digest,
    prompt_text,
    row_id,
    safe_file_stem,
    score_image_latents,
    select_prompt_rows,
    split_name,
    write_json,
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
    return patch


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
    injection_mode: str,
) -> dict[str, Any]:
    """构造统一 baseline observation。"""

    detection_decision = bool(float(score) >= float(threshold))
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
            "prompt_id": row_id(row, index, "prompt_id", "prompt"),
            "prompt_text": prompt_text(row),
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
            "latent_shape": list(latent_shape),
            "execution_device": execution_device,
            "shallow_injection_mode": injection_mode,
            "quality_score_proxy": 1.0,
            "score_retention_proxy": 1.0,
        }
    )


def generate_watermarked_image(
    pipe: Any,
    prompt: str,
    *,
    latents: Any,
    mask: Any,
    patch: Any,
    args: argparse.Namespace,
) -> tuple[Any, str]:
    """在浅层 denoising 位置注入 watermark 并生成图像。"""

    injection_step = max(0, min(int(args.num_inference_steps) - 1, int(round(float(args.edit_fraction) * int(args.num_inference_steps)))))
    injection_mode = "callback_on_step_end"
    injected_flag = {"value": False}

    def callback_on_step_end(_pipe: Any, step_index: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        if int(step_index) == injection_step:
            callback_kwargs["latents"] = inject_watermark(
                callback_kwargs["latents"],
                mask,
                patch,
                injection=str(args.w_injection),
            )
            injected_flag["value"] = True
        return callback_kwargs

    try:
        image = pipe(
            prompt,
            guidance_scale=float(args.guidance_scale),
            num_inference_steps=int(args.num_inference_steps),
            height=int(args.height),
            width=int(args.width),
            latents=latents.clone(),
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
        if injected_flag["value"]:
            return image, injection_mode
    except TypeError:
        pass

    injected_latents = inject_watermark(latents.clone(), mask, patch, injection=str(args.w_injection))
    image = pipe(
        prompt,
        guidance_scale=float(args.guidance_scale),
        num_inference_steps=int(args.num_inference_steps),
        height=int(args.height),
        width=int(args.width),
        latents=injected_latents,
    ).images[0]
    return image, "initial_latent_fallback"


def score_image(pipe: Any, image: Any, *, size: int, device: str, mask: Any, patch: Any, measurement: str, num_inversion_steps: int) -> float:
    """把图像重新编码、反演并计算 masked patch 距离分数。"""

    reversed_latents = score_image_latents(pipe, image, size=int(size), device=device, num_inversion_steps=int(num_inversion_steps))
    return score_latents(reversed_latents, mask=mask, patch=patch, measurement=measurement)


def run_shallow_diffuse_method_faithful_adapter(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行 Shallow Diffuse SD3.5 方法忠实适配流程。"""

    import torch
    from PIL import Image

    prompt_rows = select_prompt_rows(load_prompt_rows(args.prompt_plan), args.max_samples)
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    if args.require_cuda and device != "cuda":
        raise RuntimeError("shallow_diffuse_method_faithful_sd35_requires_cuda")

    pipe = load_sd3_pipeline(
        model_id=args.model_id,
        device=device,
        torch_dtype_name=args.torch_dtype,
        adapter_class_name="ShallowDiffuseInversionStableDiffusion3Pipeline",
    )
    artifact_root = Path(args.artifact_root) if args.artifact_root else Path(args.out).resolve().parent / "artifacts"
    clean_dir = artifact_root / "images" / "clean"
    watermarked_dir = artifact_root / "images" / "watermarked"
    attacked_dir = artifact_root / "images" / "attacked"
    clean_dir.mkdir(parents=True, exist_ok=True)
    watermarked_dir.mkdir(parents=True, exist_ok=True)
    attacked_dir.mkdir(parents=True, exist_ok=True)

    latent_shape = (1, int(args.latent_channels), max(1, int(args.height) // 8), max(1, int(args.width) // 8))
    observations_without_threshold: list[dict[str, Any]] = []
    image_pairs: list[dict[str, Any]] = []
    runtime_keys: dict[str, dict[str, Any]] = {}
    injection_modes: set[str] = set()

    for index, row in enumerate(prompt_rows, start=1):
        current_prompt = prompt_text(row)
        prompt_id = row_id(row, index, "prompt_id", "prompt")
        image_id = row_id(row, index, "image_id", "shallow_diffuse_image")
        file_stem = safe_file_stem(image_id, f"shallow_diffuse_image_{index:05d}")
        latent_generator = torch.Generator(device=device).manual_seed(int(args.seed) + index - 1)
        patch_generator = torch.Generator(device=device).manual_seed(int(args.watermark_seed) + index - 1)

        clean_latents = torch.randn(latent_shape, generator=latent_generator, device=device, dtype=pipe.transformer.dtype)
        mask = build_watermark_mask(
            latent_shape,
            mask_shape=str(args.w_mask_shape),
            radius=int(args.w_radius),
            inner_radius=int(args.w_inner_radius),
            channel=int(args.w_channel),
            device=device,
        )
        patch = build_watermark_patch(
            latent_shape,
            pattern=str(args.w_pattern),
            radius=int(args.w_radius),
            generator=patch_generator,
            device=device,
        )

        clean_image = pipe(
            current_prompt,
            guidance_scale=float(args.guidance_scale),
            num_inference_steps=int(args.num_inference_steps),
            height=int(args.height),
            width=int(args.width),
            latents=clean_latents,
        ).images[0]
        watermarked_image, injection_mode = generate_watermarked_image(
            pipe,
            current_prompt,
            latents=clean_latents,
            mask=mask,
            patch=patch,
            args=args,
        )
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
            num_inversion_steps=int(args.num_inversion_steps),
        )
        watermarked_score = score_image(
            pipe,
            watermarked_image,
            size=int(args.height),
            device=device,
            mask=mask,
            patch=patch,
            measurement=str(args.w_measurement),
            num_inversion_steps=int(args.num_inversion_steps),
        )

        runtime_keys[image_id] = {"mask": mask, "patch": patch, "row": row, "row_index": index, "injection_mode": injection_mode}
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
                "latent_shape": list(latent_shape),
                "shallow_injection_mode": injection_mode,
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
                injection_mode=injection_mode,
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
                injection_mode=injection_mode,
            )
        )
        if device == "cuda":
            torch.cuda.empty_cache()

    threshold, threshold_source = derive_threshold(observations_without_threshold, args.threshold)
    observations: list[dict[str, Any]] = []
    for row in observations_without_threshold:
        updated = dict(row)
        updated["threshold"] = float(threshold)
        updated["threshold_source"] = threshold_source
        updated["detection_decision"] = bool(float(updated["score"]) >= float(threshold))
        updated["final_decision"] = updated["detection_decision"]
        updated["baseline_observation_digest"] = build_stable_digest(updated)
        observations.append(updated)

    attacked_records: list[dict[str, Any]] = []
    attack_families = [item.strip() for item in str(args.attack_families or "").split(",") if item.strip()]
    for attack_family in attack_families:
        attack_matrix_family = canonical_attack_family(attack_family)
        attack_matrix_name = canonical_attack_name(attack_family)
        for pair_index, pair in enumerate(image_pairs, start=1):
            image_id = str(pair["image_id"])
            runtime = runtime_keys[image_id]
            for role_name, source_path_field, source_digest_field, sample_role in (
                ("clean", "clean_image_path", "clean_image_digest", "attacked_negative"),
                ("watermarked", "watermarked_image_path", "watermarked_image_digest", "attacked_positive"),
            ):
                with Image.open(pair[source_path_field]) as source_image:
                    attacked_image, attack_transform_name = apply_image_attack(
                        source_image,
                        attack_family=attack_family,
                        seed=int(args.seed) + pair_index,
                    )
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
                    num_inversion_steps=int(args.num_inversion_steps),
                )
                observations.append(
                    build_observation(
                        event_id=attacked_id,
                        score=score,
                        threshold=threshold,
                        threshold_source=threshold_source,
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
                        injection_mode=str(runtime["injection_mode"]),
                    )
                )
                attacked_records.append(
                    {
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
                        "attack_transform_name": attack_transform_name,
                    }
                )
                if device == "cuda":
                    torch.cuda.empty_cache()

    image_pairs_path = write_json(artifact_root / "shallow_diffuse_image_pairs.json", image_pairs)
    attacked_manifest_path = write_json(
        artifact_root / "attacked_image_manifest.json",
        {"attacked_images": attacked_records, "attacked_image_count": len(attacked_records)},
    )
    manifest = {
        "artifact_name": "shallow_diffuse_method_faithful_sd35_adapter_manifest.json",
        "producer_id": "shallow_diffuse_method_faithful_sd35_adapter",
        "baseline_id": BASELINE_ID,
        "adapter_boundary": METHOD_FAITHFUL_ADAPTER_BOUNDARY,
        "adapter_status": "method_faithful_sd35_adapter_ready",
        "model_id": args.model_id,
        "prompt_plan_path": str(Path(args.prompt_plan)),
        "baseline_observations_path": str(Path(args.out)),
        "artifact_root": str(artifact_root),
        "image_pairs_path": str(image_pairs_path),
        "attacked_image_manifest_path": str(attacked_manifest_path),
        "image_pair_count": len(image_pairs),
        "attacked_image_count": len(attacked_records),
        "observation_count": len(observations),
        "latent_shape": list(latent_shape),
        "execution_device": device,
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
            "watermark_seed": int(args.watermark_seed),
        },
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "formal_result_claim": False,
        "supports_paper_claim": False,
    }
    manifest["adapter_digest"] = build_stable_digest(manifest)
    return observations, manifest


def build_parser() -> argparse.ArgumentParser:
    """构造 Shallow Diffuse 方法忠实适配器参数解析器。"""

    parser = argparse.ArgumentParser(description="运行 Shallow Diffuse SD3.5 方法忠实 external baseline adapter")
    parser.add_argument("--prompt-plan", required=True, help="共同 prompt 计划 JSON 路径")
    parser.add_argument("--out", required=True, help="baseline observations JSON 输出路径")
    parser.add_argument("--artifact-root", default=None, help="图像、攻击结果和 manifest 输出目录")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--num-inversion-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
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
    parser.add_argument("--attack-families", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def run_cli(argv: list[str] | None = None) -> None:
    """命令行入口, 写出 observations 与 manifest。"""

    args = build_parser().parse_args(argv)
    observations, manifest = run_shallow_diffuse_method_faithful_adapter(args)
    output_path = write_json(args.out, observations)
    manifest["baseline_observations_path"] = str(output_path)
    manifest_path = Path(args.out).with_name("shallow_diffuse_method_faithful_sd35_adapter_manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    run_cli()
