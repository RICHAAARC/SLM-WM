"""定义正式图像攻击矩阵的唯一配置来源。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable, Mapping

from main.core.digest import build_stable_digest


DIFFUSION_ATTACK_INFERENCE_STEPS = 28
DIFFUSION_ATTACK_GUIDANCE_SCALE = 4.5
DIFFUSION_ATTACK_NEGATIVE_PROMPT = "low quality, blurry"
FORMAL_ATTACK_SEED_PROTOCOL = "sha256_generation_seed_attack_id_uint63_v1"


def formal_attack_seed_protocol_record() -> dict[str, Any]:
    """返回跨方法统一的攻击随机 seed 派生协议."""

    payload = {
        "formal_attack_seed_protocol": FORMAL_ATTACK_SEED_PROTOCOL,
        "input_fields": ["generation_seed_random", "attack_id"],
        "derivation": "sha256_domain_separated_first_uint64_mask_uint63",
        "output_field": "attack_seed_random",
    }
    return {
        **payload,
        "formal_attack_seed_protocol_digest": build_stable_digest(payload),
    }


def formal_attack_seed_random(
    generation_seed_random: int,
    attack_id: str,
) -> int:
    """从基础生成 seed 与攻击 ID 派生跨方法一致的非负63位 seed."""

    if type(generation_seed_random) is not int or generation_seed_random < 0:
        raise ValueError("generation_seed_random 必须是非负整数")
    normalized_attack_id = str(attack_id).strip()
    if not normalized_attack_id:
        raise ValueError("attack_id 不能为空")
    payload = (
        f"{FORMAL_ATTACK_SEED_PROTOCOL}\0{generation_seed_random}\0"
        f"{normalized_attack_id}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


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


ATTACK_RECORD_DIGEST_FIELDS = (
    "run_id",
    "prompt_id",
    "split",
    "sample_role",
    "attack_id",
    "attack_family",
    "attack_name",
    "resource_profile",
    "attack_config_digest",
    "attack_seed_random",
    "formal_attack_seed_protocol_digest",
    "measurement_digest",
    "source_image_digest",
    "attacked_image_digest",
    "frozen_threshold_digest",
    "formal_evidence_positive",
)


def build_attack_record_digest(record: Mapping[str, Any]) -> str:
    """构造绑定 Prompt、攻击身份、图像与检测结果的正式记录摘要.

    该函数是攻击记录 producer 与最终门禁共用的摘要原语.集中维护字段集合
    可以避免写出端和复验端分别实现摘要逻辑, 并防止只改 Prompt 或样本角色后
    沿用旧摘要冒充完整 test 集证据.
    """

    return build_stable_digest(
        {
            field_name: record.get(field_name)
            for field_name in ATTACK_RECORD_DIGEST_FIELDS
        }
    )


def build_attack_matrix_manifest_config(
    *,
    paper_run_name: str,
    evaluation_boundary: Mapping[str, Any],
    attack_configs: Iterable[AttackConfig],
    attack_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """构造攻击矩阵 manifest 的唯一配置摘要载荷."""

    formal_configs = tuple(
        config
        for config in attack_configs
        if config.enabled
        and config.resource_profile in {"full_main", "full_extra"}
    )
    return {
        "paper_run_name": str(paper_run_name),
        "evaluation_boundary": dict(evaluation_boundary),
        "attack_config_digest": build_stable_digest(
            [config.to_dict() for config in formal_configs]
        ),
        "attack_record_digest": build_stable_digest(
            [dict(record) for record in attack_records]
        ),
        **formal_attack_seed_protocol_record(),
    }


def resolve_formal_attack_config(
    *,
    attack_family: str,
    attack_name: str,
    resource_profile: str | None = None,
) -> AttackConfig:
    """从唯一正式攻击注册表解析完整配置.

    该函数属于通用协议写法: observation producer, 正式导入器和结果 schema
    共用同一解析入口, 从而避免仅按攻击名称后贴资源档位或配置摘要.
    """

    candidates = tuple(
        config
        for config in default_attack_configs()
        if config.enabled
        and config.resource_profile in {"full_main", "full_extra"}
        and config.attack_family == str(attack_family)
        and config.attack_name == str(attack_name)
        and (
            resource_profile is None
            or config.resource_profile == str(resource_profile)
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            "正式攻击配置必须唯一: "
            f"{attack_family}/{attack_name}/{resource_profile or '*'}"
        )
    return candidates[0]


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
            attack_parameters={
                "denoise_strength": 0.35,
                "inference_steps": DIFFUSION_ATTACK_INFERENCE_STEPS,
                "guidance_scale": DIFFUSION_ATTACK_GUIDANCE_SCALE,
                "negative_prompt": DIFFUSION_ATTACK_NEGATIVE_PROMPT,
            },
        ),
        AttackConfig(
            attack_id="flow_matching_inversion_regeneration_extra",
            attack_family="regeneration_attack",
            attack_name="flow_matching_inversion_regeneration",
            attack_strength=0.40,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={
                "inversion_steps": 30,
                "reconstruction_steps": 30,
                "guidance_scale": DIFFUSION_ATTACK_GUIDANCE_SCALE,
                "negative_prompt": DIFFUSION_ATTACK_NEGATIVE_PROMPT,
            },
        ),
        AttackConfig(
            attack_id="sdedit_regeneration_extra",
            attack_family="regeneration_attack",
            attack_name="sdedit_regeneration",
            attack_strength=0.45,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={
                "noise_level": 0.45,
                "inference_steps": DIFFUSION_ATTACK_INFERENCE_STEPS,
                "guidance_scale": DIFFUSION_ATTACK_GUIDANCE_SCALE,
                "negative_prompt": DIFFUSION_ATTACK_NEGATIVE_PROMPT,
            },
        ),
        AttackConfig(
            attack_id="diffusion_purification_extra",
            attack_family="regeneration_attack",
            attack_name="diffusion_purification",
            attack_strength=0.32,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={
                "purification_steps": 20,
                "noise_level": 0.32,
                "guidance_scale": 1.0,
                "negative_prompt": "",
            },
        ),
        AttackConfig(
            attack_id="global_editing_extra",
            attack_family="global_editing_attack",
            attack_name="global_editing_attack",
            attack_strength=0.48,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={
                "denoise_strength": 0.48,
                "edit_prompt_suffix": "with a changed global style and lighting",
                "inference_steps": DIFFUSION_ATTACK_INFERENCE_STEPS,
                "guidance_scale": DIFFUSION_ATTACK_GUIDANCE_SCALE,
                "negative_prompt": DIFFUSION_ATTACK_NEGATIVE_PROMPT,
            },
        ),
        AttackConfig(
            attack_id="local_editing_extra",
            attack_family="local_editing_attack",
            attack_name="local_editing_attack",
            attack_strength=0.42,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={
                "denoise_strength": 0.42,
                "local_mask_ratio": 0.36,
                "edit_prompt_suffix": "with the central subject locally changed while the surrounding scene remains unchanged",
                "inference_steps": DIFFUSION_ATTACK_INFERENCE_STEPS,
                "guidance_scale": DIFFUSION_ATTACK_GUIDANCE_SCALE,
                "negative_prompt": DIFFUSION_ATTACK_NEGATIVE_PROMPT,
            },
        ),
        AttackConfig(
            attack_id="visual_paraphrase_extra",
            attack_family="visual_paraphrase_attack",
            attack_name="visual_paraphrase_attack",
            attack_strength=0.55,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={
                "denoise_strength": 0.55,
                "paraphrase_prompt_suffix": "redrawn with the same semantics but different visual composition",
                "inference_steps": DIFFUSION_ATTACK_INFERENCE_STEPS,
                "guidance_scale": DIFFUSION_ATTACK_GUIDANCE_SCALE,
                "negative_prompt": DIFFUSION_ATTACK_NEGATIVE_PROMPT,
            },
        ),
        AttackConfig(
            attack_id="adversarial_removal_extra",
            attack_family="adversarial_removal_attack",
            attack_name="adversarial_removal_attack",
            attack_strength=0.55,
            resource_profile="full_extra",
            requires_gpu=True,
            enabled=True,
            attack_parameters={
                "candidate_query_count": 8,
                "strength_min": 0.25,
                "strength_max": 0.55,
                "inference_steps": DIFFUSION_ATTACK_INFERENCE_STEPS,
                "guidance_scale": DIFFUSION_ATTACK_GUIDANCE_SCALE,
                "negative_prompt": DIFFUSION_ATTACK_NEGATIVE_PROMPT,
            },
        ),
    )
