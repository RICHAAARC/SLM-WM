"""SD3.5 主表外部扩散 baseline 方法忠实适配公共工具。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from main.core.digest import build_stable_digest

METHOD_FAITHFUL_ADAPTER_BOUNDARY = "method_faithful_sd35_adapter_reproduction"
FORMAL_IMAGE_ATTACK_SPECS = {
    "jpeg": ("standard_distortion", "jpeg_compression"),
    "jpeg_compression": ("standard_distortion", "jpeg_compression"),
    "gaussian_noise": ("standard_distortion", "gaussian_noise"),
    "noise": ("standard_distortion", "gaussian_noise"),
    "gaussian_blur": ("standard_distortion", "gaussian_blur"),
    "blur": ("standard_distortion", "gaussian_blur"),
    "rotation": ("geometric_transform", "rotation"),
    "rotate": ("geometric_transform", "rotation"),
    "resize": ("geometric_transform", "resize"),
    "crop": ("geometric_transform", "crop"),
    "crop_resize": ("geometric_transform", "crop_resize"),
    "composite_geometric_attacks": ("geometric_transform", "composite_geometric_attacks"),
}
FORMAL_IMAGE_ATTACK_NAMES = (
    "jpeg_compression",
    "gaussian_noise",
    "gaussian_blur",
    "rotation",
    "resize",
    "crop",
    "crop_resize",
    "composite_geometric_attacks",
)


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
    """计算文件 SHA-256 摘要, 用于图像 provenance 与 governed import。"""

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
        text = as_text(row.get("prompt_text") or row.get("prompt") or row.get("caption") or row.get("text"))
        if not text:
            raise ValueError(f"prompt plan 第 {index} 行缺少 prompt 文本字段")
        normalized.append(dict(row))
    if not normalized:
        raise ValueError("prompt plan 不能为空")
    return normalized


def select_prompt_rows(rows: list[dict[str, Any]], max_samples: int | None) -> list[dict[str, Any]]:
    """按可选样本上限截取 prompt rows。"""

    if max_samples is None:
        return rows
    return rows[: max(0, int(max_samples))]


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
    """生成中心圆形 mask。"""

    import numpy as np

    x0 = int(size) // 2 + int(x_offset)
    y0 = int(size) // 2 + int(y_offset)
    y_axis, x_axis = np.ogrid[: int(size), : int(size)]
    y_axis = y_axis[::-1]
    return ((x_axis - x0) ** 2 + (y_axis - y0) ** 2) <= int(radius) ** 2


def normalize_attack_request(attack_family: str) -> str:
    """把外部传入的攻击名称规范化为攻击矩阵使用的语义名称。"""

    text = str(attack_family).strip().lower().replace("-", "_").replace(" ", "_")
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    if text not in FORMAL_IMAGE_ATTACK_SPECS:
        raise ValueError(f"unsupported_sd35_adapter_attack:{attack_family}")
    return FORMAL_IMAGE_ATTACK_SPECS[text][1]


def canonical_attack_family(attack_family: str) -> str:
    """返回攻击矩阵共同协议中的攻击族名称。"""

    return FORMAL_IMAGE_ATTACK_SPECS[normalize_attack_request(attack_family)][0]


def canonical_attack_name(attack_family: str) -> str:
    """返回攻击矩阵共同协议中的攻击名称。"""

    return FORMAL_IMAGE_ATTACK_SPECS[normalize_attack_request(attack_family)][1]


def supported_formal_image_attack_names() -> tuple[str, ...]:
    """返回当前 method-faithful adapter 可生成的正式图像级攻击名称集合。"""

    return FORMAL_IMAGE_ATTACK_NAMES


class InversionStableDiffusion3PipelineMixin:
    """为 StableDiffusion3Pipeline 增加外部 baseline 检测所需的轻量反演方法。"""

    def get_image_latents(self, image: Any, *, sample: bool = False) -> Any:
        """通过 VAE 编码图像得到 latent。"""

        import torch

        with torch.inference_mode():
            encoding_dist = self.vae.encode(image).latent_dist
            encoding = encoding_dist.sample() if sample else encoding_dist.mode()
            shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0) or 0.0)
            scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0) or 1.0)
            return (encoding - shift_factor) * scaling_factor

    def approximate_forward_diffusion(
        self,
        latents: Any,
        *,
        prompt: str = "",
        num_inference_steps: int = 5,
        guidance_scale: float = 1.0,
    ) -> Any:
        """使用 SD3 scheduler 近似执行从图像 latent 到噪声 latent 的反演。"""

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
            for index, timestep in enumerate(reversed(timesteps)):
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
                latents = latents - (self.scheduler.sigmas[index + 1] - self.scheduler.sigmas[index]) * noise_pred
                del noise_pred, latent_model_input, timestep_tensor
            return latents


def load_sd3_pipeline(*, model_id: str, device: str, torch_dtype_name: str, adapter_class_name: str) -> Any:
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
    pipeline_class = type(adapter_class_name, (InversionStableDiffusion3PipelineMixin, StableDiffusion3Pipeline), {})
    pipe = pipeline_class.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


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


def score_image_latents(pipe: Any, image: Any, *, size: int, device: str, num_inversion_steps: int) -> Any:
    """把图像重新编码并近似反演为检测 latent。"""

    tensor = image_to_tensor(image, size=int(size), device=device, dtype=pipe.vae.dtype)
    image_latents = pipe.get_image_latents(tensor, sample=False)
    return pipe.approximate_forward_diffusion(
        image_latents,
        prompt="",
        num_inference_steps=int(num_inversion_steps),
        guidance_scale=1.0,
    )


def derive_threshold(observations: Iterable[dict[str, Any]], explicit_threshold: float | None) -> tuple[float, str]:
    """从 calibration split 或当前 clean 样本派生阈值。"""

    if explicit_threshold is not None:
        return float(explicit_threshold), "cli_threshold"
    rows = list(observations)
    calibration_rows = [row for row in rows if row.get("split") == "calibration"] or rows
    negative_scores = [float(row["score"]) for row in calibration_rows if row.get("sample_role") == "clean_negative"]
    positive_scores = [float(row["score"]) for row in calibration_rows if row.get("sample_role") == "positive_source"]
    if negative_scores and positive_scores:
        return (max(negative_scores) + min(positive_scores)) / 2.0, "midpoint_between_negative_max_and_positive_min"
    return 0.0, "fallback_zero_insufficient_calibration_pairs"


def apply_image_attack(image: Any, *, attack_family: str, seed: int) -> tuple[Any, str]:
    """对 PIL 图像执行轻量图像级攻击。"""

    from io import BytesIO
    import random
    import numpy as np
    from PIL import Image, ImageFilter

    family = normalize_attack_request(attack_family)
    rng = random.Random(int(seed))
    source = image.convert("RGB")
    width, height = source.size

    def center_crop(fraction: float) -> Any:
        crop_width = max(1, int(width * float(fraction)))
        crop_height = max(1, int(height * float(fraction)))
        left = max(0, (width - crop_width) // 2)
        upper = max(0, (height - crop_height) // 2)
        return source.crop((left, upper, left + crop_width, upper + crop_height))

    if family == "jpeg_compression":
        buffer = BytesIO()
        source.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB"), "jpeg_quality_75"
    if family == "gaussian_noise":
        array = np.asarray(source).astype("float32")
        noise_rng = np.random.default_rng(int(seed))
        noisy = np.clip(array + noise_rng.normal(0.0, 8.0, size=array.shape), 0, 255).astype("uint8")
        return Image.fromarray(noisy, mode="RGB"), "gaussian_noise_sigma_8"
    if family == "gaussian_blur":
        return source.filter(ImageFilter.GaussianBlur(radius=1.0)), "gaussian_blur_radius_1"
    if family == "rotation":
        angle = 5.0 if rng.random() >= 0.5 else -5.0
        return source.rotate(angle, resample=Image.Resampling.BICUBIC), f"rotation_{angle:g}_degree"
    if family == "resize":
        resized = source.resize((max(1, int(width * 0.75)), max(1, int(height * 0.75))), Image.Resampling.BICUBIC)
        return resized.resize((width, height), Image.Resampling.BICUBIC), "resize_downscale_0.75_restore"
    if family == "crop":
        return center_crop(0.90), "center_crop_0.90"
    if family == "crop_resize":
        return center_crop(0.85).resize((width, height), Image.Resampling.BICUBIC), "center_crop_0.85_resize"
    if family == "composite_geometric_attacks":
        angle = 5.0 if rng.random() >= 0.5 else -5.0
        rotated = source.rotate(angle, resample=Image.Resampling.BICUBIC)
        crop_width = max(1, int(width * 0.90))
        crop_height = max(1, int(height * 0.90))
        left = max(0, (width - crop_width) // 2)
        upper = max(0, (height - crop_height) // 2)
        cropped = rotated.crop((left, upper, left + crop_width, upper + crop_height))
        return cropped.resize((width, height), Image.Resampling.BICUBIC), f"rotation_{angle:g}_degree_crop_resize"
    raise ValueError(f"unsupported_sd35_adapter_attack:{attack_family}")


def observation_digest(payload: dict[str, Any]) -> dict[str, Any]:
    """为 observation 补齐稳定摘要。"""

    row = dict(payload)
    row["baseline_observation_digest"] = build_stable_digest(row)
    return row
