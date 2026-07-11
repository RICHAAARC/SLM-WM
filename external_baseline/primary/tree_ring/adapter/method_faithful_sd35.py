"""Tree-Ring 在 SD3.5 Medium 上的方法忠实适配器。

该模块只位于 external baseline 适配层, 不属于 SLM-WM 主方法实现。它参考
Tree-Ring 官方源码的核心机制: 在扩散初始 latent 的傅里叶域中心环形区域写入 key,
生成图像后通过图像到 latent 的流匹配反向 Euler 积分恢复初始噪声, 再用 key 距离进行检测。

通用工程写法:
- Notebook 或命令计划只调用本模块入口, 不直接手写 records。
- 真实模型加载、图像落盘、observation 构造和 manifest 写出集中在适配器边界。

项目特定写法:
- SD3.5 Medium 使用 16-channel latent, 因此这里把原 Tree-Ring 的 4-channel latent
  显式推广到可配置通道数。
- 输出 observation 标记为 method-faithful adapter, 但仍默认不声明论文主张。是否能进入
  主表正式结果由 formal import validator 依据 当前论文运行层级的完整 Prompt、fixed-FPR 和攻击矩阵边界决定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from experiments.runtime.image_metrics import measured_image_ssim, measured_score_retention
from main.core.digest import build_stable_digest
from external_baseline.primary.sd35_method_faithful_common import (
    apply_formal_image_attack,
    canonical_attack_family,
    canonical_attack_name,
    emit_adapter_progress,
)

BASELINE_ID = "tree_ring"
ADAPTER_BOUNDARY = "method_faithful_sd35_adapter_reproduction"
DEFAULT_SCORE_NAME = "negative_tree_ring_fft_key_distance"


def load_json(path: str | Path) -> Any:
    """读取 JSON 文件, 兼容 Windows 和 Colab 常见 BOM。"""

    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, payload: Any) -> Path:
    """写出稳定 JSON 文件并返回路径。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def file_digest(path: str | Path) -> str:
    """计算文件 SHA256 摘要, 用于图像 provenance 与 governed import。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_text(value: Any, default: str = "") -> str:
    """把可选字段规范化为非空字符串。"""

    if value is None:
        return default
    text = str(value).strip()
    return text or default


def load_prompt_rows(path: str | Path) -> list[dict[str, Any]]:
    """读取共同 prompt plan, 支持 list 或包含 prompts/items/records/prompt_rows 的 object。"""

    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for field_name in ("prompts", "items", "records", "prompt_rows"):
            candidate = payload.get(field_name)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            raise ValueError("prompt plan object 必须包含 prompts/items/records/prompt_rows 列表字段")
    else:
        raise TypeError("prompt plan 必须是 JSON list 或 object")

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"prompt plan 第 {index} 行必须是 object")
        prompt_text = as_text(row.get("prompt_text") or row.get("prompt") or row.get("caption") or row.get("text"))
        if not prompt_text:
            raise ValueError(f"prompt plan 第 {index} 行缺少 prompt 文本字段")
        normalized.append(dict(row))
    if not normalized:
        raise ValueError("prompt plan 不能为空")
    return normalized


def safe_file_stem(value: str, fallback: str) -> str:
    """把图像标识转换为安全文件名主干。"""

    candidate = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value).strip("._")
    return candidate or fallback


def row_id(row: dict[str, Any], index: int, field_name: str, fallback_prefix: str) -> str:
    """读取行标识, 缺失时用稳定序号补齐。"""

    return as_text(row.get(field_name), f"{fallback_prefix}_{index:05d}")


def prompt_text(row: dict[str, Any]) -> str:
    """读取 prompt 文本。"""

    return as_text(row.get("prompt_text") or row.get("prompt") or row.get("caption") or row.get("text"), "unspecified prompt")


def split_name(row: dict[str, Any]) -> str:
    """读取数据 split, 缺失时归入 test。"""

    return as_text(row.get("split"), "test")


def circle_mask(size: int, radius: int, *, x_offset: int = 0, y_offset: int = 0) -> Any:
    """生成 Tree-Ring 使用的傅里叶域圆形中心 mask。"""

    import numpy as np

    x0 = size // 2 + x_offset
    y0 = size // 2 + y_offset
    y_axis, x_axis = np.ogrid[:size, :size]
    y_axis = y_axis[::-1]
    return ((x_axis - x0) ** 2 + (y_axis - y0) ** 2) <= radius**2


def build_watermark_key(shape: tuple[int, int, int, int], *, pattern: str, radius: int, generator: Any, device: str) -> Any:
    """构造 Tree-Ring 傅里叶域 key。

    此处属于方法忠实适配: ring/rand/zeros 三种 pattern 与官方源码保持同一语义,
    但 latent shape 由 SD3.5 参数决定。
    """

    import torch

    init = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    patch = torch.fft.fftshift(torch.fft.fft2(init), dim=(-1, -2))
    if "zeros" in pattern:
        patch = patch * 0
    elif "ring" in pattern:
        source = patch.clone().detach()
        latent_size = int(shape[-1])
        for current_radius in range(int(radius), 0, -1):
            ring_mask = torch.tensor(circle_mask(latent_size, current_radius), device=device, dtype=torch.bool)
            for channel in range(int(shape[1])):
                patch[:, channel, ring_mask] = source[0, channel, 0, current_radius].item()
    elif "rand" in pattern:
        patch[:] = patch[0]
    else:
        raise ValueError(f"unsupported_tree_ring_pattern:{pattern}")
    return patch


def build_watermark_mask(shape: tuple[int, int, int, int], *, channel: int, radius: int, device: str) -> Any:
    """构造 Tree-Ring 写入区域 mask。"""

    import torch

    if channel < -1 or channel >= shape[1]:
        raise ValueError(f"w_channel 必须在 [-1, {shape[1] - 1}] 范围内")
    latent_size = int(shape[-1])
    torch_mask = torch.tensor(circle_mask(latent_size, int(radius)), device=device, dtype=torch.bool)
    mask = torch.zeros(shape, device=device, dtype=torch.bool)
    if channel == -1:
        mask[:, :, torch_mask] = True
    else:
        mask[:, channel, torch_mask] = True
    return mask


def inject_watermark(latents: Any, mask: Any, key: Any) -> Any:
    """在 latent 傅里叶域写入 Tree-Ring key。"""

    import torch

    latents_fft = torch.fft.fftshift(torch.fft.fft2(latents.float()), dim=(-1, -2))
    latents_fft[mask] = key[mask].clone()
    injected = torch.fft.ifft2(torch.fft.ifftshift(latents_fft, dim=(-1, -2))).real
    return injected.to(dtype=latents.dtype)


def score_latents(reversed_latents: Any, mask: Any, key: Any) -> float:
    """计算检测分数。

    Tree-Ring 原始距离越小越像阳性。本项目 observation 默认使用 higher_is_positive=True,
    因此返回负距离, 使分数越大越像阳性。
    """

    import torch

    reversed_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents.float()), dim=(-1, -2))
    distance = torch.abs(reversed_fft[mask] - key[mask]).mean().item()
    return -float(distance)


def image_to_tensor(image: Any, *, size: int, device: str, dtype: Any) -> Any:
    """把 PIL 图像转换为 SD VAE 输入张量, 范围为 [-1, 1]。"""

    import numpy as np
    import torch
    from PIL import Image

    resized = image.convert("RGB").resize((int(size), int(size)), Image.Resampling.BICUBIC)
    array = np.asarray(resized).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device=device, dtype=dtype)


class InversionStableDiffusion3PipelineMixin:
    """为 StableDiffusion3Pipeline 增加 Tree-Ring 检测所需的真实流匹配反演方法。"""

    def get_image_latents(self, image: Any, *, sample: bool = False) -> Any:
        """通过 VAE 编码图像得到 latent。"""

        import torch

        with torch.inference_mode():
            encoding_dist = self.vae.encode(image).latent_dist
            encoding = encoding_dist.sample() if sample else encoding_dist.mode()
            shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0) or 0.0)
            scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0) or 1.0)
            return (encoding - shift_factor) * scaling_factor

    def invert_flow_matching_latent(
        self,
        latents: Any,
        *,
        prompt: str = "",
        num_inference_steps: int = 5,
        guidance_scale: float = 1.0,
    ) -> Any:
        """使用 SD3 scheduler 迭代执行从图像 latent 到初始噪声 latent 的反演。

        该函数是适配器内的数值积分, 主要用于让 Tree-Ring 的检测路径在 SD3.5 Medium 上可审计。
        官方原始环境仍通过 legacy DDIM inversion 作为补充表忠实度参考。
        """

        import torch

        with torch.inference_mode():
            self.scheduler.set_timesteps(int(num_inference_steps), device=self._execution_device)
            do_classifier_free_guidance = float(guidance_scale) > 1.0
            prompt_embeds, _, pooled_projections, _ = self.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                prompt_3=None,
                device=self._execution_device,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            timesteps = self.scheduler.timesteps
            sigmas = self.scheduler.sigmas
            for schedule_index in range(len(timesteps) - 1, -1, -1):
                timestep = timesteps[schedule_index]
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                timestep_tensor = timestep.expand(latent_model_input.shape[0])
                noise_pred = self.transformer(
                    latent_model_input,
                    timestep=timestep_tensor,
                    pooled_projections=pooled_projections,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + float(guidance_scale) * (noise_pred_text - noise_pred_uncond)
                sigma_current = sigmas[schedule_index + 1].to(
                    device=latents.device,
                    dtype=latents.dtype,
                )
                sigma_next = sigmas[schedule_index].to(
                    device=latents.device,
                    dtype=latents.dtype,
                )
                latents = latents + (sigma_next - sigma_current) * noise_pred
                del noise_pred, latent_model_input, timestep_tensor
            return latents


def load_sd3_pipeline(*, model_id: str, device: str, torch_dtype_name: str) -> Any:
    """加载 SD3.5 pipeline 并动态组合反演 mixin。"""

    import torch
    from diffusers import StableDiffusion3Pipeline

    dtype_lookup = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_lookup.get(str(torch_dtype_name).lower(), torch.float16)
    pipeline_class = type(
        "TreeRingInversionStableDiffusion3Pipeline",
        (InversionStableDiffusion3PipelineMixin, StableDiffusion3Pipeline),
        {},
    )
    pipe = pipeline_class.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


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
    quality_score: float,
    score_retention: float,
) -> dict[str, Any]:
    """构造统一 baseline observation。"""

    detection_decision = bool(float(score) >= float(threshold))
    payload = {
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
        "producer_id": "tree_ring_method_faithful_sd35_adapter",
        "producer_role": "external_baseline_method_faithful_adapter",
        "adapter_boundary": ADAPTER_BOUNDARY,
        "threshold_source": threshold_source,
        "formal_result_claim": False,
        "supports_paper_claim": False,
        "generation_model_id": model_id,
        "latent_shape": list(latent_shape),
        "execution_device": execution_device,
        "quality_score": float(quality_score),
        "score_retention": float(score_retention),
    }
    payload["baseline_observation_digest"] = build_stable_digest(payload)
    return payload


def derive_threshold(
    observations: Iterable[dict[str, Any]],
    explicit_threshold: float | None,
    target_fpr: float,
) -> tuple[float, str]:
    """仅从 calibration clean negative 分数冻结 fixed-FPR 阈值。"""

    if explicit_threshold is not None:
        return float(explicit_threshold), "pre_registered_threshold"
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr 必须位于 (0, 1)")
    rows = list(observations)
    negative_scores = [
        float(row["score"])
        for row in rows
        if row.get("split") == "calibration" and row.get("sample_role") == "clean_negative"
    ]
    if not negative_scores:
        raise ValueError("fixed-FPR 阈值要求非空 calibration clean negative 分数")
    allowed_false_positives = max(
        0,
        math.floor(target_fpr * (len(negative_scores) + 1)) - 1,
    )
    for threshold in sorted({math.nextafter(score, math.inf) for score in negative_scores}):
        if sum(score >= threshold for score in negative_scores) <= allowed_false_positives:
            return threshold, "calibration_clean_negative_conformal"
    raise RuntimeError("无法从 calibration clean negative 冻结 fixed-FPR 阈值")



def score_image(pipe: Any, image: Any, *, size: int, device: str, mask: Any, key: Any, num_inversion_steps: int) -> float:
    """把图像重新编码并计算 Tree-Ring 检测分数。"""

    tensor = image_to_tensor(image, size=int(size), device=device, dtype=pipe.vae.dtype)
    image_latents = pipe.get_image_latents(tensor, sample=False)
    reversed_latents = pipe.invert_flow_matching_latent(
        image_latents,
        prompt="",
        num_inference_steps=int(num_inversion_steps),
        guidance_scale=1.0,
    )
    return score_latents(reversed_latents, mask, key)


def run_tree_ring_method_faithful_adapter(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行 Tree-Ring SD3.5 方法忠实适配流程。"""

    import torch
    from PIL import Image

    prompt_rows = load_prompt_rows(args.prompt_plan)
    if args.max_samples is not None:
        prompt_rows = prompt_rows[: max(0, int(args.max_samples))]
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
        raise RuntimeError("tree_ring_method_faithful_sd35_requires_cuda")

    pipe = load_sd3_pipeline(model_id=args.model_id, device=device, torch_dtype_name=args.torch_dtype)
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
        image_id = row_id(row, index, "image_id", "tree_ring_image")
        file_stem = safe_file_stem(image_id, f"tree_ring_image_{index:05d}")
        generator = torch.Generator(device=device).manual_seed(int(args.seed) + index - 1)
        key_generator = torch.Generator(device=device).manual_seed(int(args.watermark_seed) + index - 1)

        clean_latents = torch.randn(latent_shape, generator=generator, device=device, dtype=pipe.transformer.dtype)
        key = build_watermark_key(
            latent_shape,
            pattern=args.w_pattern,
            radius=int(args.w_radius),
            generator=key_generator,
            device=device,
        )
        mask = build_watermark_mask(latent_shape, channel=int(args.w_channel), radius=int(args.w_radius), device=device)
        watermarked_latents = inject_watermark(clean_latents.clone(), mask, key)

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
        watermarked_path = watermarked_dir / f"{file_stem}_tree_ring.png"
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
            key=key,
            num_inversion_steps=int(args.num_inversion_steps),
        )
        watermarked_score = score_image(
            pipe,
            watermarked_image,
            size=int(args.height),
            device=device,
            mask=mask,
            key=key,
            num_inversion_steps=int(args.num_inversion_steps),
        )
        pair_quality = measured_image_ssim(clean_image, watermarked_image)

        runtime_keys[image_id] = {
            "mask": mask,
            "key": key,
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
                            mask=runtime["mask"],
                            key=runtime["key"],
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
                    mask=runtime["mask"],
                    key=runtime["key"],
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
                        quality_score=attack_quality,
                        score_retention=measured_score_retention(
                            float(runtime[f"{role_name}_score"]),
                            score,
                        ),
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
                        "attack_execution": attack_execution,
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

    image_pairs_path = write_json(artifact_root / "tree_ring_image_pairs.json", image_pairs)
    attacked_manifest_path = write_json(
        artifact_root / "attacked_image_manifest.json",
        {"attacked_images": attacked_records, "attacked_image_count": len(attacked_records)},
    )
    manifest = {
        "artifact_name": "tree_ring_method_faithful_sd35_adapter_manifest.json",
        "producer_id": "tree_ring_method_faithful_sd35_adapter",
        "baseline_id": BASELINE_ID,
        "adapter_boundary": ADAPTER_BOUNDARY,
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
        "generation_protocol": {
            "model_id": args.model_id,
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
            "w_channel": int(args.w_channel),
            "w_radius": int(args.w_radius),
            "w_pattern": args.w_pattern,
            "watermark_seed": int(args.watermark_seed),
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
    """构造 Tree-Ring 方法忠实适配器参数解析器。"""

    parser = argparse.ArgumentParser(description="运行 Tree-Ring SD3.5 方法忠实 external baseline adapter")
    parser.add_argument("--prompt-plan", required=True, help="共同 prompt 计划 JSON 路径")
    parser.add_argument("--out", required=True, help="baseline observations JSON 输出路径")
    parser.add_argument("--artifact-root", default=None, help="图像、攻击结果和 manifest 输出目录")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--num-inversion-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watermark-seed", type=int, default=999999)
    parser.add_argument("--w-channel", type=int, default=0)
    parser.add_argument("--w-radius", type=int, default=10)
    parser.add_argument("--w-pattern", default="ring", choices=("ring", "rand", "zeros"))
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
    observations, manifest = run_tree_ring_method_faithful_adapter(args)
    output_path = write_json(args.out, observations)
    manifest["baseline_observations_path"] = str(output_path)
    manifest_path = Path(args.out).with_name("tree_ring_method_faithful_sd35_adapter_manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    run_cli()
