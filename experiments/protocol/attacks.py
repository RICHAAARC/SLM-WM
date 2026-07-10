"""定义正式图像攻击矩阵的唯一配置来源。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from main.core.digest import build_stable_digest


@dataclass(frozen=True)
class AttackConfig:
    """描述单个攻击配置。

    该配置属于通用工程写法: 将攻击名称、强度、资源档位和参数集中到
    dataclass 构造时校验, 使后续记录构建函数只表达攻击矩阵的数据流。
    """

    attack_id: str
    attack_family: str
    attack_name: str
    attack_strength: float
    resource_profile: str
    requires_gpu: bool
    enabled: bool
    attack_parameters: dict[str, Any]

    def __post_init__(self) -> None:
        """集中校验攻击配置边界。"""
        if self.attack_strength < 0.0:
            raise ValueError("attack_strength 不得小于 0")
        if self.resource_profile not in {"probe", "full_main", "full_extra"}:
            raise ValueError("resource_profile 必须属于受治理资源档位")
        if not self.attack_id or not self.attack_name or not self.attack_family:
            raise ValueError("攻击配置标识、名称和族名称不得为空")

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


def attack_config_digest(config: AttackConfig) -> str:
    """生成攻击配置的稳定摘要。"""
    return build_stable_digest(config.to_dict())


def default_attack_configs() -> tuple[AttackConfig, ...]:
    """返回默认攻击矩阵配置。"""
    return (
        AttackConfig(
            attack_id="jpeg_compression_probe",
            attack_family="standard_distortion",
            attack_name="jpeg_compression",
            attack_strength=0.10,
            resource_profile="probe",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"quality": 90},
        ),
        AttackConfig(
            attack_id="jpeg_compression_main",
            attack_family="standard_distortion",
            attack_name="jpeg_compression",
            attack_strength=0.35,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"quality": 60},
        ),
        AttackConfig(
            attack_id="gaussian_noise_main",
            attack_family="standard_distortion",
            attack_name="gaussian_noise",
            attack_strength=0.30,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"sigma": 0.03},
        ),
        AttackConfig(
            attack_id="gaussian_blur_main",
            attack_family="standard_distortion",
            attack_name="gaussian_blur",
            attack_strength=0.28,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"radius": 1.2},
        ),
        AttackConfig(
            attack_id="resize_main",
            attack_family="geometric_transform",
            attack_name="resize",
            attack_strength=0.25,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"scale": 0.75},
        ),
        AttackConfig(
            attack_id="crop_main",
            attack_family="geometric_transform",
            attack_name="crop",
            attack_strength=0.30,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"crop_ratio": 0.82},
        ),
        AttackConfig(
            attack_id="rotation_main",
            attack_family="geometric_transform",
            attack_name="rotation",
            attack_strength=0.24,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"degrees": 5.0},
        ),
        AttackConfig(
            attack_id="crop_resize_main",
            attack_family="geometric_transform",
            attack_name="crop_resize",
            attack_strength=0.34,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"crop_ratio": 0.80, "resize_scale": 1.0},
        ),
        AttackConfig(
            attack_id="composite_geometric_main",
            attack_family="geometric_transform",
            attack_name="composite_geometric_attacks",
            attack_strength=0.42,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"crop_ratio": 0.78, "degrees": 7.0, "resize_scale": 0.85},
        ),
        AttackConfig(
            attack_id="photometric_distortion_main",
            attack_family="photometric_distortion_attack",
            attack_name="photometric_distortion_attack",
            attack_strength=0.30,
            resource_profile="full_main",
            requires_gpu=False,
            enabled=True,
            attack_parameters={"brightness": 1.12, "contrast": 1.10, "saturation": 0.88, "gamma": 0.92},
        ),
        AttackConfig(
            attack_id="img2img_regeneration_extra",
            attack_family="regeneration_attack",
            attack_name="img2img_regeneration",
            attack_strength=0.35,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={"denoise_strength": 0.35},
        ),
        AttackConfig(
            attack_id="ddim_inversion_regeneration_extra",
            attack_family="regeneration_attack",
            attack_name="ddim_inversion_regeneration",
            attack_strength=0.40,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={"inversion_steps": 30, "denoise_strength": 0.40},
        ),
        AttackConfig(
            attack_id="sdedit_regeneration_extra",
            attack_family="regeneration_attack",
            attack_name="sdedit_regeneration",
            attack_strength=0.45,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={"noise_level": 0.45},
        ),
        AttackConfig(
            attack_id="diffusion_purification_extra",
            attack_family="regeneration_attack",
            attack_name="diffusion_purification",
            attack_strength=0.32,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={"purification_steps": 20, "noise_level": 0.32},
        ),
        AttackConfig(
            attack_id="global_editing_extra",
            attack_family="global_editing_attack",
            attack_name="global_editing_attack",
            attack_strength=0.48,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={"denoise_strength": 0.48, "edit_prompt_suffix": "with a changed global style and lighting"},
        ),
        AttackConfig(
            attack_id="local_editing_extra",
            attack_family="local_editing_attack",
            attack_name="local_editing_attack",
            attack_strength=0.42,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={"denoise_strength": 0.42, "local_mask_ratio": 0.36},
        ),
        AttackConfig(
            attack_id="visual_paraphrase_extra",
            attack_family="visual_paraphrase_attack",
            attack_name="visual_paraphrase_attack",
            attack_strength=0.55,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={"denoise_strength": 0.55, "paraphrase_prompt_suffix": "redrawn with the same semantics but different visual composition"},
        ),
        AttackConfig(
            attack_id="adversarial_removal_extra",
            attack_family="adversarial_removal_attack",
            attack_name="adversarial_removal_attack",
            attack_strength=0.55,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={"query_count": 8, "strength_min": 0.25, "strength_max": 0.55},
        ),
    )
