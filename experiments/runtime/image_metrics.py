"""定义主方法与外部 baseline 共享的图像质量和分数保持率。"""

from __future__ import annotations

import math
from typing import Any


def image_to_rgb_tensor(image: Any) -> Any:
    """把 PIL 图像转换为范围为 [0, 1] 的 NCHW RGB tensor。"""

    import numpy as np
    import torch

    array = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def measured_image_ssim(reference_image: Any, candidate_image: Any) -> float:
    """使用最大11x11高斯窗口计算两幅 RGB 图像的标准 SSIM。"""

    import torch
    import torch.nn.functional as functional

    if candidate_image.size != reference_image.size:
        raise ValueError("SSIM 要求两幅图像具有相同尺寸")
    window_size = min(11, *reference_image.size)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        raise ValueError("SSIM 要求图像的最短边至少为3像素")
    reference = image_to_rgb_tensor(reference_image)
    candidate = image_to_rgb_tensor(candidate_image)
    axis = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel_1d = torch.exp(-(axis.square()) / (2.0 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = (kernel_1d[:, None] @ kernel_1d[None, :]).expand(
        3,
        1,
        window_size,
        window_size,
    )
    mean_reference = functional.conv2d(reference, kernel_2d, groups=3)
    mean_candidate = functional.conv2d(candidate, kernel_2d, groups=3)
    variance_reference = (
        functional.conv2d(reference.square(), kernel_2d, groups=3)
        - mean_reference.square()
    )
    variance_candidate = (
        functional.conv2d(candidate.square(), kernel_2d, groups=3)
        - mean_candidate.square()
    )
    covariance = (
        functional.conv2d(reference * candidate, kernel_2d, groups=3)
        - mean_reference * mean_candidate
    )
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2.0 * mean_reference * mean_candidate + c1)
        * (2.0 * covariance + c2)
        / (
            (mean_reference.square() + mean_candidate.square() + c1)
            * (variance_reference + variance_candidate + c2)
        )
    )
    return float(score.mean().item())


def measured_score_retention(source_score: float, evaluated_score: float) -> float:
    """计算单一检测器内部的分数稳定性诊断值。

    该值依赖检测器自身分数尺度，只用于同一方法内部排查攻击前后数值漂移，
    不得作为跨方法正式比较指标。跨方法鲁棒性只使用冻结 fixed-FPR 下的 TPR/FPR 与置信区间。
    """

    source = float(source_score)
    evaluated = float(evaluated_score)
    if not math.isfinite(source) or not math.isfinite(evaluated):
        raise ValueError("分数保持率要求有限数值")
    scale = max(abs(source), 1e-6)
    return math.exp(-abs(evaluated - source) / scale)


def compute_image_quality_metrics(
    reference_image: Any,
    candidate_image: Any,
) -> dict[str, float | str]:
    """计算主方法与 baseline 共用的 paired image 质量指标。"""

    if candidate_image.size != reference_image.size:
        raise ValueError("成对图像质量指标要求两幅图像具有相同尺寸")
    reference = image_to_rgb_tensor(reference_image)
    candidate = image_to_rgb_tensor(candidate_image)
    difference = reference - candidate
    mse = float(difference.square().mean().item())
    mean_abs_error = float(difference.abs().mean().item())
    psnr: float | str = "inf" if mse == 0.0 else float(
        20.0 * math.log10(1.0 / math.sqrt(mse))
    )
    return {
        "psnr": psnr,
        "ssim": measured_image_ssim(reference_image, candidate_image),
        "mse": mse,
        "mean_abs_error": mean_abs_error,
    }
