"""定义 SD runtime adapter 的通用协议对象。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from main.core.digest import build_stable_digest

from experiments.runtime.diffusion.latent_estimator import estimate_image_digest, estimate_quality_score, vector_digest


@dataclass(frozen=True)
class RuntimeModelConfig:
    """描述一次 SD runtime probe 的最小配置。"""

    model_family: str
    model_id: str
    backend_mode: str
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    inference_steps: int
    guidance_scale: float
    latent_width: int = 8

    def __post_init__(self) -> None:
        """集中校验配置边界, 避免业务路径重复防御式校验。"""
        positive_int_fields = {
            "width": self.width,
            "height": self.height,
            "inference_steps": self.inference_steps,
            "latent_width": self.latent_width,
        }
        invalid_fields = {name: value for name, value in positive_int_fields.items() if value <= 0}
        if invalid_fields:
            raise ValueError(f"配置正整数边界无效: {invalid_fields}")
        if self.guidance_scale <= 0.0:
            raise ValueError("guidance_scale 必须为正数")

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)

    def build_run_id(self) -> str:
        """根据配置生成稳定 run_id。"""
        return f"{self.model_family}_{build_stable_digest(self.to_dict())[:16]}"


@dataclass(frozen=True)
class RuntimeProbeBundle:
    """保存一次 runtime adapter probe 的全部 records。"""

    generation_record: dict[str, Any]
    latent_trace_records: tuple[Any, ...]
    attention_capture_records: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return {
            "generation_record": self.generation_record,
            "latent_trace_records": [record.to_dict() for record in self.latent_trace_records],
            "attention_capture_records": [record.to_dict() for record in self.attention_capture_records],
        }


class DiffusionRuntimeAdapter(Protocol):
    """SD runtime adapter 需要实现的最小接口。"""

    def generate(self, config: RuntimeModelConfig) -> RuntimeProbeBundle:
        """运行一次 generation probe 并返回 records。"""


def build_generation_record(
    config: RuntimeModelConfig,
    run_id: str,
    backend_name: str,
    runtime_dependency_mode: str,
    final_latent: tuple[float, ...],
    unsupported_reason: str,
) -> dict[str, Any]:
    """把最终 latent 和配置转换为 generation record。"""
    prompt_digest = build_stable_digest(
        {
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "model_id": config.model_id,
            "seed": config.seed,
        }
    )
    latent_digest = vector_digest(final_latent)
    return {
        "generation_id": f"generation_{run_id}",
        "run_id": run_id,
        "model_family": config.model_family,
        "model_id": config.model_id,
        "backend_name": backend_name,
        "backend_mode": config.backend_mode,
        "runtime_dependency_mode": runtime_dependency_mode,
        "prompt_digest": prompt_digest,
        "seed": config.seed,
        "width": config.width,
        "height": config.height,
        "inference_steps": config.inference_steps,
        "guidance_scale": config.guidance_scale,
        "latent_digest": latent_digest,
        "image_digest": estimate_image_digest(final_latent, config.model_id, config.seed),
        "image_shape": (config.height, config.width, 3),
        "quality_score": estimate_quality_score(final_latent),
        "unsupported_reason": unsupported_reason,
        "metadata": {
            "records_are_synthetic": True,
            "supports_paper_claim": False,
        },
    }
