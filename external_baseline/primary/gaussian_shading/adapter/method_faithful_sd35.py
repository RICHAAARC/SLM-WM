"""Gaussian Shading 在 SD3.5 Medium 上的方法忠实适配器。

该模块位于 external baseline 适配层。它保留 Gaussian Shading 的核心机制:
使用二值 message 控制 latent noise 的正负截断 Gaussian 采样, 生成图像后通过
图像编码和流匹配反向 Euler 积分恢复 noise sign, 再经 key 解码与 block voting 得到检测分数。

项目特定写法:
- SD3.5 Medium 使用 16-channel latent, 因此 message 与 watermark 的重复映射从
  官方 4-channel latent 显式推广到可配置通道数。
- observation 标记为 method-faithful adapter, 但仍不直接声明论文主张。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.runtime.image_metrics import measured_image_ssim, measured_score_retention
from experiments.protocol.attacks import attack_config_digest
from main.core.digest import build_stable_digest

from external_baseline.primary.sd35_method_faithful_common import (
    DEFAULT_SD35_MODEL_ID,
    DEFAULT_SD35_MODEL_REVISION,
    METHOD_FAITHFUL_ADAPTER_BOUNDARY,
    apply_formal_image_attack,
    canonical_attack_family,
    canonical_attack_name,
    derive_threshold,
    emit_adapter_progress,
    file_digest,
    formal_image_attack_config,
    load_prompt_rows,
    load_sd3_pipeline,
    observation_digest,
    prompt_text,
    row_id,
    safe_file_stem,
    score_image_latents,
    select_prompt_rows,
    split_name,
    validate_model_revision,
    validated_observation_attack_identity,
    write_json,
)

BASELINE_ID = "gaussian_shading"
DEFAULT_SCORE_NAME = "gaussian_shading_bit_vote_accuracy"


class GaussianShadingWatermark:
    """SD3.5 latent 形状下的 Gaussian Shading message 与 voting 状态。"""

    def __init__(
        self,
        *,
        latent_shape: tuple[int, int, int, int],
        channel_copy: int,
        hw_copy: int,
        generator: Any,
        device: str,
    ) -> None:
        """初始化 watermark、key 与重复映射参数。"""

        import torch

        batch, channels, height, width = latent_shape
        if batch != 1:
            raise ValueError("gaussian_shading_sd35_adapter_currently_requires_batch_one")
        if channels % int(channel_copy) != 0 or height % int(hw_copy) != 0 or width % int(hw_copy) != 0:
            raise ValueError("gaussian_shading_copy_factors_must_divide_sd35_latent_shape")
        self.latent_shape = latent_shape
        self.channel_copy = int(channel_copy)
        self.hw_copy = int(hw_copy)
        self.device = device
        self.key = torch.randint(0, 2, latent_shape, generator=generator, device=device, dtype=torch.int64)
        self.watermark = torch.randint(
            0,
            2,
            (batch, channels // self.channel_copy, height // self.hw_copy, width // self.hw_copy),
            generator=generator,
            device=device,
            dtype=torch.int64,
        )
        self.vote_threshold = max(1, self.channel_copy * self.hw_copy * self.hw_copy // 2)

    def expanded_watermark(self) -> Any:
        """把低维 watermark bit 重复到完整 SD3.5 latent 形状。"""

        return (
            self.watermark.repeat_interleave(self.channel_copy, dim=1)
            .repeat_interleave(self.hw_copy, dim=2)
            .repeat_interleave(self.hw_copy, dim=3)
        )

    def create_watermarked_latents(self, *, dtype: Any, generator: Any) -> Any:
        """按 Gaussian Shading 的截断 Gaussian message 采样水印 latent。"""

        import torch

        message = (self.expanded_watermark() + self.key) % 2
        magnitude = torch.clamp(torch.abs(torch.randn(self.latent_shape, generator=generator, device=self.device)), min=1e-4)
        signed_latents = (message.to(dtype=torch.float32) * 2.0 - 1.0) * magnitude
        return signed_latents.to(dtype=dtype)

    def create_clean_latents(self, *, dtype: Any, generator: Any) -> Any:
        """生成未携带 Gaussian Shading message 的 clean latent。"""

        import torch

        return torch.randn(self.latent_shape, generator=generator, device=self.device, dtype=dtype)

    def decode_recovered_watermark(self, reversed_latents: Any) -> Any:
        """从反演 latent 的 sign 恢复 watermark bit 并执行 block voting。"""

        import torch

        reversed_message = (reversed_latents.float() > 0).to(dtype=torch.int64)
        decoded = (reversed_message + self.key) % 2
        batch, channels, height, width = self.latent_shape
        split_dim1 = torch.cat(torch.split(decoded, channels // self.channel_copy, dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, height // self.hw_copy, dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, width // self.hw_copy, dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        return (vote > self.vote_threshold).to(dtype=torch.int64).unsqueeze(0)

    def score_latents(self, reversed_latents: Any) -> float:
        """计算 recovered watermark 与原 watermark 的 bit accuracy。"""

        recovered = self.decode_recovered_watermark(reversed_latents)
        return float((recovered == self.watermark).float().mean().detach().cpu().item())


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
            "image_id": image_id,
            "image_path": image_path,
            "image_digest": image_digest,
            "producer_id": "gaussian_shading_method_faithful_sd35_adapter",
            "producer_role": "external_baseline_method_faithful_adapter",
            "adapter_boundary": METHOD_FAITHFUL_ADAPTER_BOUNDARY,
            "threshold_source": threshold_source,
            "formal_result_claim": False,
            "supports_paper_claim": False,
            "generation_model_id": model_id,
            "generation_model_revision": validate_model_revision(model_revision),
            "latent_shape": list(latent_shape),
            "execution_device": execution_device,
            "quality_score": float(quality_score),
            "score_retention": float(score_retention),
        }
    )


def score_image(pipe: Any, image: Any, *, size: int, device: str, watermark: GaussianShadingWatermark, num_inversion_steps: int) -> float:
    """把图像重新编码、反演并计算 Gaussian Shading bit vote 分数。"""

    reversed_latents = score_image_latents(pipe, image, size=int(size), device=device, num_inversion_steps=int(num_inversion_steps))
    return watermark.score_latents(reversed_latents)


def run_gaussian_shading_method_faithful_adapter(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行 Gaussian Shading SD3.5 方法忠实适配流程。"""

    import torch
    from PIL import Image

    prompt_rows = select_prompt_rows(load_prompt_rows(args.prompt_plan), args.max_samples)
    attack_families = [item.strip() for item in str(args.attack_families or "").split(",") if item.strip()]
    progress_total = max(1, 1 + len(prompt_rows) + len(attack_families) * len(prompt_rows) * 2)
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
        raise RuntimeError("gaussian_shading_method_faithful_sd35_requires_cuda")

    pipe = load_sd3_pipeline(
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=device,
        torch_dtype_name=args.torch_dtype,
        adapter_class_name="GaussianShadingInversionStableDiffusion3Pipeline",
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
    observations_without_threshold: list[dict[str, Any]] = []
    image_pairs: list[dict[str, Any]] = []
    runtime_keys: dict[str, dict[str, Any]] = {}

    for index, row in enumerate(prompt_rows, start=1):
        current_prompt = prompt_text(row)
        prompt_id = row_id(row, index, "prompt_id", "prompt")
        image_id = row_id(row, index, "image_id", "gaussian_shading_image")
        file_stem = safe_file_stem(image_id, f"gaussian_shading_image_{index:05d}")
        latent_generator = torch.Generator(device=device).manual_seed(int(args.seed) + index - 1)
        watermark_generator = torch.Generator(device=device).manual_seed(int(args.watermark_seed) + index - 1)
        sampling_generator = torch.Generator(device=device).manual_seed(int(args.watermark_seed) + 100000 + index - 1)

        watermark = GaussianShadingWatermark(
            latent_shape=latent_shape,
            channel_copy=int(args.channel_copy),
            hw_copy=int(args.hw_copy),
            generator=watermark_generator,
            device=device,
        )
        clean_latents = watermark.create_clean_latents(dtype=pipe.transformer.dtype, generator=latent_generator)
        watermarked_latents = watermark.create_watermarked_latents(dtype=pipe.transformer.dtype, generator=sampling_generator)

        clean_image = pipe(
            current_prompt,
            guidance_scale=float(args.guidance_scale),
            num_inference_steps=int(args.num_inference_steps),
            height=int(args.height),
            width=int(args.width),
            latents=clean_latents,
        ).images[0]
        watermarked_image = pipe(
            current_prompt,
            guidance_scale=float(args.guidance_scale),
            num_inference_steps=int(args.num_inference_steps),
            height=int(args.height),
            width=int(args.width),
            latents=watermarked_latents,
        ).images[0]

        clean_path = clean_dir / f"{file_stem}_clean.png"
        watermarked_path = watermarked_dir / f"{file_stem}_gaussian_shading.png"
        clean_image.save(clean_path)
        watermarked_image.save(watermarked_path)
        clean_digest = file_digest(clean_path)
        watermarked_digest = file_digest(watermarked_path)
        clean_score = score_image(
            pipe,
            clean_image,
            size=int(args.height),
            device=device,
            watermark=watermark,
            num_inversion_steps=int(args.num_inversion_steps),
        )
        watermarked_score = score_image(
            pipe,
            watermarked_image,
            size=int(args.height),
            device=device,
            watermark=watermark,
            num_inversion_steps=int(args.num_inversion_steps),
        )
        pair_quality = measured_image_ssim(clean_image, watermarked_image)

        runtime_keys[image_id] = {
            "watermark": watermark,
            "row": row,
            "row_index": index,
            "clean_score": clean_score,
            "watermarked_score": watermarked_score,
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
                quality_score=pair_quality,
                score_retention=1.0,
            )
        )
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
    for attack_family in attack_families:
        formal_attack_config = formal_image_attack_config(attack_family)
        attack_matrix_family = canonical_attack_family(attack_family)
        attack_matrix_name = canonical_attack_name(attack_family)
        formal_attack_digest = attack_config_digest(formal_attack_config)
        for pair_index, pair in enumerate(image_pairs, start=1):
            image_id = str(pair["image_id"])
            runtime = runtime_keys[image_id]
            for role_name, source_path_field, source_digest_field, sample_role in (
                ("clean", "clean_image_path", "clean_image_digest", "attacked_negative"),
                ("watermarked", "watermarked_image_path", "watermarked_image_digest", "attacked_positive"),
            ):
                with Image.open(pair[source_path_field]) as source_image:
                    attacked_image, attack_transform_name, attack_execution = apply_formal_image_attack(
                        source_image,
                        attack_family=attack_family,
                        seed=int(args.seed) + pair_index,
                        pipe=pipe,
                        prompt=str(pair["prompt_text"]),
                        size=int(args.height),
                        device=device,
                        detection_score=lambda candidate: score_image(
                            pipe,
                            candidate,
                            size=int(args.height),
                            device=device,
                            watermark=runtime["watermark"],
                            num_inversion_steps=int(args.num_inversion_steps),
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
                    watermark=runtime["watermark"],
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
                        model_revision=args.model_revision,
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
                        "attack_id": formal_attack_config.attack_id,
                        "resource_profile": formal_attack_config.resource_profile,
                        "attack_config_digest": formal_attack_digest,
                        "attack_transform_name": attack_transform_name,
                        "attack_execution": attack_execution,
                        "generation_model_id": args.model_id,
                        "generation_model_revision": args.model_revision,
                    }
                )
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

    image_pairs_path = write_json(artifact_root / "gaussian_shading_image_pairs.json", image_pairs)
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
        "artifact_name": "gaussian_shading_method_faithful_sd35_adapter_manifest.json",
        "producer_id": "gaussian_shading_method_faithful_sd35_adapter",
        "baseline_id": BASELINE_ID,
        "adapter_boundary": METHOD_FAITHFUL_ADAPTER_BOUNDARY,
        "adapter_status": "method_faithful_sd35_adapter_ready",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
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
        "generation_protocol": {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "height": int(args.height),
            "width": int(args.width),
        },
        "detection_protocol": {
            "input_access_mode": "image_only",
            "num_inversion_steps": int(args.num_inversion_steps),
            "target_fpr": float(args.target_fpr),
        },
        "watermark_parameters": {
            "channel_copy": int(args.channel_copy),
            "hw_copy": int(args.hw_copy),
            "watermark_seed": int(args.watermark_seed),
            "message_mapping": "sign_truncated_gaussian_with_repeated_bit_voting",
        },
        "threshold": float(threshold),
        "threshold_source": threshold_source,
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
    """构造 Gaussian Shading 方法忠实适配器参数解析器。"""

    parser = argparse.ArgumentParser(description="运行 Gaussian Shading SD3.5 方法忠实 external baseline adapter")
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
    parser.add_argument("--watermark-seed", type=int, default=20260622)
    parser.add_argument("--channel-copy", type=int, default=1)
    parser.add_argument("--hw-copy", type=int, default=8)
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
    observations, manifest = run_gaussian_shading_method_faithful_adapter(args)
    output_path = write_json(args.out, observations)
    manifest["baseline_observations_path"] = str(output_path)
    manifest_path = Path(args.out).with_name("gaussian_shading_method_faithful_sd35_adapter_manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    run_cli()
