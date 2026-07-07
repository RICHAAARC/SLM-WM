"""SD3/SD3.5 pipeline 运行时通用工具。

该模块属于核心方法复现层, 用于把真实模型加载、tensor 摘要辅助和轻量图像质量
计算从 Notebook helper 中抽离出来。这样主方法 runner 可以在服务器和 Colab
入口中复用同一套实现, 不需要反向依赖 `paper_workflow/`。
"""

from __future__ import annotations

import math
import os
from typing import Any

from experiments.runtime.repository_environment import build_runtime_environment_report, flatten_environment_versions


def import_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    """延迟导入真实模型和图像依赖。"""

    import torch
    import diffusers
    from diffusers import StableDiffusion3Pipeline

    return None, torch, diffusers, StableDiffusion3Pipeline


def load_pipeline(config: Any) -> tuple[Any, dict[str, Any]]:
    """加载真实 SD3 系列 pipeline 并移动到目标设备。

    `config` 只要求提供 `device_name`、`torch_dtype`、`hf_token_env` 和
    `model_id` 字段, 因而可被最小机制预检 runner 与正式 latent injection
    runner 共同复用。
    """

    _, torch, _, pipeline_class = import_runtime_dependencies()
    if config.device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gpu_unavailable")
    dtype = getattr(torch, config.torch_dtype)
    token = os.environ.get(config.hf_token_env) or None
    pipeline = pipeline_class.from_pretrained(config.model_id, torch_dtype=dtype, token=token)
    pipeline = pipeline.to(config.device_name)
    pipeline.set_progress_bar_config(disable=False)
    environment_report = build_runtime_environment_report(torch_module=torch)
    runtime_versions = {
        **flatten_environment_versions(environment_report),
        "runtime_environment": environment_report,
    }
    return pipeline, runtime_versions


def tensor_norm(tensor: Any) -> float:
    """计算 tensor 的二范数。"""

    return float(tensor.detach().float().norm().item())


def compute_image_quality_metrics(clean_image: Any, watermarked_image: Any) -> dict[str, float | str]:
    """计算 paired image 的轻量质量指标。"""

    import torch

    def _image_tensor(image: Any) -> Any:
        """将 PIL 图像转成 HWC float tensor, 避免质量指标路径依赖 NumPy。"""

        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        image_bytes = bytearray(rgb_image.tobytes())
        return torch.frombuffer(image_bytes, dtype=torch.uint8).reshape(height, width, 3).float() / 255.0

    clean = _image_tensor(clean_image)
    watermarked = _image_tensor(watermarked_image)
    diff = clean - watermarked
    mse = float((diff * diff).mean().item())
    mean_abs_error = float(diff.abs().mean().item())
    psnr: float | str = "inf" if mse == 0.0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    clean_mean = float(clean.mean().item())
    watermarked_mean = float(watermarked.mean().item())
    clean_var = float(clean.var(unbiased=False).item())
    watermarked_var = float(watermarked.var(unbiased=False).item())
    covariance = float(((clean - clean_mean) * (watermarked - watermarked_mean)).mean().item())
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = float(
        ((2 * clean_mean * watermarked_mean + c1) * (2 * covariance + c2))
        / ((clean_mean**2 + watermarked_mean**2 + c1) * (clean_var + watermarked_var + c2))
    )
    return {"psnr": psnr, "ssim": ssim, "mse": mse, "mean_abs_error": mean_abs_error}
