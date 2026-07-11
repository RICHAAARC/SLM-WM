"""对真实 PIL 图像执行可复现的标准失真与几何攻击。"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from experiments.protocol.attacks import AttackConfig


def _center_crop(image: Any, ratio: float) -> Any:
    """按比例执行中心裁剪。"""

    width, height = image.size
    crop_width = max(1, round(width * ratio))
    crop_height = max(1, round(height * ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def apply_standard_image_attack(image: Any, config: AttackConfig, seed: int) -> Any:
    """执行协议中不需要生成模型的真实图像攻击。"""

    import torch
    from PIL import Image, ImageEnhance, ImageFilter

    if config.requires_gpu:
        raise ValueError("该函数只处理不依赖生成模型的图像攻击")
    rgb = image.convert("RGB")
    width, height = rgb.size
    parameters = config.attack_parameters
    if config.attack_name == "jpeg_compression":
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=int(parameters["quality"]))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").copy()
    if config.attack_name == "gaussian_noise":
        pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).reshape(height, width, 3).float() / 255.0
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(pixels.shape, generator=generator) * float(parameters["sigma"])
        attacked = ((pixels + noise).clamp(0.0, 1.0) * 255.0).byte().numpy()
        return Image.fromarray(attacked, mode="RGB")
    if config.attack_name == "gaussian_blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=float(parameters["radius"])))
    if config.attack_name == "resize":
        scale = float(parameters["scale"])
        small = rgb.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BICUBIC)
        return small.resize((width, height), Image.Resampling.BICUBIC)
    if config.attack_name == "crop":
        return _center_crop(rgb, float(parameters["crop_ratio"])).resize((width, height), Image.Resampling.BICUBIC)
    if config.attack_name == "rotation":
        return rgb.rotate(float(parameters["degrees"]), resample=Image.Resampling.BICUBIC, expand=False)
    if config.attack_name == "crop_resize":
        cropped = _center_crop(rgb, float(parameters["crop_ratio"]))
        resize_scale = float(parameters["resize_scale"])
        resized = cropped.resize(
            (
                max(1, round(width * resize_scale)),
                max(1, round(height * resize_scale)),
            ),
            Image.Resampling.BICUBIC,
        )
        return resized.resize((width, height), Image.Resampling.BICUBIC)
    if config.attack_name == "composite_geometric_attacks":
        cropped = _center_crop(rgb, float(parameters["crop_ratio"]))
        rotated = cropped.rotate(float(parameters["degrees"]), resample=Image.Resampling.BICUBIC, expand=False)
        intermediate_scale = float(parameters["resize_scale"])
        resized = rotated.resize(
            (max(1, round(width * intermediate_scale)), max(1, round(height * intermediate_scale))),
            Image.Resampling.BICUBIC,
        )
        return resized.resize((width, height), Image.Resampling.BICUBIC)
    if config.attack_name == "photometric_distortion_attack":
        attacked = ImageEnhance.Brightness(rgb).enhance(float(parameters["brightness"]))
        attacked = ImageEnhance.Contrast(attacked).enhance(float(parameters["contrast"]))
        attacked = ImageEnhance.Color(attacked).enhance(float(parameters["saturation"]))
        gamma = float(parameters["gamma"])
        table = [round(((value / 255.0) ** gamma) * 255.0) for value in range(256)]
        return attacked.point(table * 3)
    raise ValueError(f"尚未实现标准图像攻击: {config.attack_name}")
