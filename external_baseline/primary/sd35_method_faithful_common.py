"""SD3.5 主表外部扩散 baseline 方法忠实适配公共工具。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from main.core.digest import build_stable_digest

METHOD_FAITHFUL_ADAPTER_BOUNDARY = "method_faithful_sd35_adapter_reproduction"
STANDARD_GEOMETRIC_FORMAL_IMAGE_ATTACK_NAMES = (
    "jpeg_compression",
    "gaussian_noise",
    "gaussian_blur",
    "rotation",
    "resize",
    "crop",
    "crop_resize",
    "composite_geometric_attacks",
    "photometric_distortion_attack",
)
REGENERATION_FORMAL_IMAGE_ATTACK_NAMES = (
    "img2img_regeneration",
    "flow_matching_inversion_regeneration",
    "sdedit_regeneration",
    "diffusion_purification",
)
ADVANCED_GPU_FORMAL_IMAGE_ATTACK_NAMES = (
    "global_editing_attack",
    "local_editing_attack",
    "visual_paraphrase_attack",
    "adversarial_removal_attack",
)
DIFFUSION_FORMAL_IMAGE_ATTACK_NAMES = REGENERATION_FORMAL_IMAGE_ATTACK_NAMES + ADVANCED_GPU_FORMAL_IMAGE_ATTACK_NAMES
FORMAL_IMAGE_ATTACK_NAMES = STANDARD_GEOMETRIC_FORMAL_IMAGE_ATTACK_NAMES + DIFFUSION_FORMAL_IMAGE_ATTACK_NAMES
FORMAL_IMAGE_ATTACK_SPECS = {
    "jpeg_compression": ("standard_distortion", "jpeg_compression"),
    "gaussian_noise": ("standard_distortion", "gaussian_noise"),
    "gaussian_blur": ("standard_distortion", "gaussian_blur"),
    "rotation": ("geometric_transform", "rotation"),
    "resize": ("geometric_transform", "resize"),
    "crop": ("geometric_transform", "crop"),
    "crop_resize": ("geometric_transform", "crop_resize"),
    "composite_geometric_attacks": ("geometric_transform", "composite_geometric_attacks"),
    "photometric_distortion_attack": ("photometric_distortion_attack", "photometric_distortion_attack"),
    "img2img_regeneration": ("regeneration_attack", "img2img_regeneration"),
    "flow_matching_inversion_regeneration": (
        "regeneration_attack",
        "flow_matching_inversion_regeneration",
    ),
    "sdedit_regeneration": ("regeneration_attack", "sdedit_regeneration"),
    "diffusion_purification": ("regeneration_attack", "diffusion_purification"),
    "global_editing_attack": ("global_editing_attack", "global_editing_attack"),
    "local_editing_attack": ("local_editing_attack", "local_editing_attack"),
    "visual_paraphrase_attack": ("visual_paraphrase_attack", "visual_paraphrase_attack"),
    "adversarial_removal_attack": ("adversarial_removal_attack", "adversarial_removal_attack"),
}


def load_json(path: str | Path) -> Any:
    """读取 JSON 文件, 兼容 Windows 和 Colab 常见 BOM。"""

    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, payload: Any) -> Path:
    """写出稳定 JSON 文件并返回路径。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def emit_adapter_progress(
    *,
    baseline_id: str,
    operation: str,
    completed: int,
    total: int,
    profile: str = "",
    **metadata: Any,
) -> None:
    """向外层 runner 回传 method-faithful adapter 细粒度进度。

    该函数属于项目通用工程写法: adapter 仍然只负责真实生成、攻击和检测,
    进度事件通过环境变量指定的 JSONL 文件传给父进程。Notebook 只显示父进程
    合并后的低噪声单行进度, 不直接承载任何正式实验逻辑。
    """

    try:
        from experiments.runtime.progress import progress_event_path_from_environment, write_progress_event

        write_progress_event(
            progress_event_path_from_environment(),
            desc="method-faithful adapter",
            completed=completed,
            total=total,
            profile=profile or f"operation={operation}",
            baseline_id=baseline_id,
            operation=operation,
            **metadata,
        )
    except Exception:
        return


def file_digest(path: str | Path) -> str:
    """计算文件 SHA-256 摘要, 用于图像 provenance 与 governed import。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measured_image_ssim(reference_image: Any, candidate_image: Any) -> float:
    """使用标准高斯窗口 SSIM 测量两幅 RGB 图像的感知一致性。"""

    import numpy as np
    import torch
    import torch.nn.functional as functional

    reference = np.asarray(reference_image.convert("RGB"), dtype="float32") / 255.0
    candidate = np.asarray(
        candidate_image.convert("RGB").resize(reference_image.size),
        dtype="float32",
    ) / 255.0
    left = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0)
    right = torch.from_numpy(candidate).permute(2, 0, 1).unsqueeze(0)
    axis = torch.arange(11, dtype=torch.float32) - 5.0
    kernel_1d = torch.exp(-(axis.square()) / (2.0 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = (kernel_1d[:, None] @ kernel_1d[None, :]).expand(3, 1, 11, 11)
    mu_left = functional.conv2d(left, kernel_2d, padding=5, groups=3)
    mu_right = functional.conv2d(right, kernel_2d, padding=5, groups=3)
    variance_left = functional.conv2d(left.square(), kernel_2d, padding=5, groups=3) - mu_left.square()
    variance_right = functional.conv2d(right.square(), kernel_2d, padding=5, groups=3) - mu_right.square()
    covariance = functional.conv2d(left * right, kernel_2d, padding=5, groups=3) - mu_left * mu_right
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = (
        (2.0 * mu_left * mu_right + c1)
        * (2.0 * covariance + c2)
        / ((mu_left.square() + mu_right.square() + c1) * (variance_left + variance_right + c2))
    )
    return float(ssim.mean().clamp(0.0, 1.0).item())


def measured_score_retention(source_score: float, evaluated_score: float) -> float:
    """把攻击前后真实检测分数变化映射为 [0, 1] 保持率。"""

    scale = max(abs(float(source_score)), 1e-6)
    return math.exp(-abs(float(evaluated_score) - float(source_score)) / scale)


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


def standard_geometric_formal_image_attack_names() -> tuple[str, ...]:
    """返回不需要额外扩散生成的常规失真与几何攻击名称集合。"""

    return STANDARD_GEOMETRIC_FORMAL_IMAGE_ATTACK_NAMES


def regeneration_formal_image_attack_names() -> tuple[str, ...]:
    """返回需要真实扩散 pipeline 参与的再生成攻击名称集合。"""

    return REGENERATION_FORMAL_IMAGE_ATTACK_NAMES


def advanced_gpu_formal_image_attack_names() -> tuple[str, ...]:
    """返回需要真实扩散 pipeline 参与的高级编辑与去水印攻击名称集合。"""

    return ADVANCED_GPU_FORMAL_IMAGE_ATTACK_NAMES


def diffusion_formal_image_attack_names() -> tuple[str, ...]:
    """返回所有需要真实扩散 pipeline 参与的图像级攻击名称集合。"""

    return DIFFUSION_FORMAL_IMAGE_ATTACK_NAMES


def is_regeneration_attack(attack_family: str) -> bool:
    """判断外部传入的攻击请求是否需要真实扩散 pipeline。"""

    return normalize_attack_request(attack_family) in DIFFUSION_FORMAL_IMAGE_ATTACK_NAMES


def formal_image_attack_resource_profile(attack_family: str) -> str:
    """返回攻击矩阵中该攻击默认对应的资源档位。"""

    return "full_extra" if is_regeneration_attack(attack_family) else "full_main"


class InversionStableDiffusion3PipelineMixin:
    """为 StableDiffusion3Pipeline 增加外部 baseline 检测所需的真实流匹配反演方法。"""

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
        """使用 SD3 scheduler 迭代执行从图像 latent 到噪声 latent 的反演。"""

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
    """把图像重新编码并流匹配反向 Euler 积分为检测 latent。"""

    tensor = image_to_tensor(image, size=int(size), device=device, dtype=pipe.vae.dtype)
    image_latents = pipe.get_image_latents(tensor, sample=False)
    return pipe.invert_flow_matching_latent(
        image_latents,
        prompt="",
        num_inference_steps=int(num_inversion_steps),
        guidance_scale=1.0,
    )


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


def apply_standard_geometric_image_attack(
    image: Any,
    *,
    attack_family: str,
    seed: int,
) -> tuple[Any, str, dict[str, Any]]:
    """复用项目共同攻击配置执行完全相同的标准图像攻击。"""

    from experiments.protocol.attacks import attack_config_digest, default_attack_configs
    from experiments.runtime.image_attacks import apply_standard_image_attack

    attack_name = normalize_attack_request(attack_family)
    if attack_name in DIFFUSION_FORMAL_IMAGE_ATTACK_NAMES:
        raise ValueError(f"regeneration_attack_requires_pipeline:{attack_family}")
    candidates = tuple(
        config
        for config in default_attack_configs()
        if config.enabled
        and not config.requires_gpu
        and config.resource_profile == "full_main"
        and config.attack_name == attack_name
    )
    if len(candidates) != 1:
        raise RuntimeError(f"共同攻击协议没有唯一配置: {attack_name}")
    config = candidates[0]
    attacked = apply_standard_image_attack(image, config, seed)
    implementation = f"shared_attack_protocol:{attack_config_digest(config)}"
    trace = {
        "attack_name": config.attack_name,
        "attack_implementation": implementation,
        "attack_seed_random": int(seed),
        "effective_parameters": dict(config.attack_parameters),
    }
    return attacked, implementation, trace

_DIFFUSION_ATTACK_RUNTIME_CACHE_ATTRIBUTE = "_slm_wm_diffusion_attack_runtime_cache"


def _shared_diffusion_attack_runtime(pipe: Any, *, size: int, device: str) -> Any:
    """为同一 SD3.5 pipeline 复用唯一共同扩散攻击运行时。"""

    from experiments.runtime.diffusion.regeneration_attacks import (
        DiffusionAttackRuntime,
        DiffusionAttackRuntimeConfig,
    )

    cache = getattr(pipe, _DIFFUSION_ATTACK_RUNTIME_CACHE_ATTRIBUTE, None)
    if cache is None:
        cache = {}
        setattr(pipe, _DIFFUSION_ATTACK_RUNTIME_CACHE_ATTRIBUTE, cache)
    cache_key = (str(device), int(size))
    if cache_key not in cache:
        config = DiffusionAttackRuntimeConfig(
            device_name=str(device),
            height=int(size),
            width=int(size),
        )
        cache[cache_key] = DiffusionAttackRuntime.from_text_to_image_pipeline(pipe, config)
    return cache[cache_key]


def apply_regeneration_image_attack(
    image: Any,
    *,
    attack_family: str,
    seed: int,
    pipe: Any,
    prompt: str,
    size: int,
    device: str,
    detection_score: Callable[[Any], float] | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    """委托项目唯一共同运行时执行真实 SD3.5 扩散攻击。"""

    from experiments.runtime.diffusion.regeneration_attacks import diffusion_attack_spec

    family = normalize_attack_request(attack_family)
    if family not in DIFFUSION_FORMAL_IMAGE_ATTACK_NAMES:
        raise ValueError(f"regeneration_attack_name_required:{attack_family}")
    spec = diffusion_attack_spec(family)
    runtime = _shared_diffusion_attack_runtime(pipe, size=int(size), device=device)
    execution = runtime.apply(
        image.convert("RGB"),
        spec,
        seed=int(seed),
        prompt_text=prompt,
        detection_score=detection_score,
    )
    return (
        execution.image.convert("RGB"),
        spec.attack_implementation,
        execution.to_record(),
    )


def apply_formal_image_attack(
    image: Any,
    *,
    attack_family: str,
    seed: int,
    pipe: Any | None = None,
    prompt: str = "",
    size: int = 512,
    device: str = "cuda",
    detection_score: Callable[[Any], float] | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    """执行共同攻击矩阵中的图像级攻击, 并对再生成攻击显式要求 pipeline。"""

    if is_regeneration_attack(attack_family):
        if pipe is None:
            raise RuntimeError(f"regeneration_attack_pipeline_missing:{attack_family}")
        return apply_regeneration_image_attack(
            image,
            attack_family=attack_family,
            seed=int(seed),
            pipe=pipe,
            prompt=prompt,
            size=int(size),
            device=device,
            detection_score=detection_score,
        )
    return apply_standard_geometric_image_attack(image, attack_family=attack_family, seed=int(seed))


def observation_digest(payload: dict[str, Any]) -> dict[str, Any]:
    """为 observation 补齐稳定摘要。"""

    row = dict(payload)
    row["baseline_observation_digest"] = build_stable_digest(row)
    return row
