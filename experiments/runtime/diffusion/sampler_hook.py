"""实现 SD 采样回调的轻量协议对象。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SamplingTrace:
    """保存一次 synthetic 采样过程中的 latent 序列。"""

    timesteps: tuple[float, ...]
    trajectory_vectors: tuple[tuple[float, ...], ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


def build_timestep_schedule(inference_steps: int) -> tuple[float, ...]:
    """构造从高噪声到低噪声的 synthetic timestep 序列。"""
    if inference_steps <= 0:
        raise ValueError("inference_steps 必须为正数")
    if inference_steps == 1:
        return (1.0,)
    return tuple(1.0 - index / (inference_steps - 1) for index in range(inference_steps))


def run_synthetic_sampler(
    initial_latent: tuple[float, ...],
    prompt_delta: tuple[float, ...],
    inference_steps: int,
    guidance_scale: float,
) -> SamplingTrace:
    """运行可复现的 synthetic diffusion 采样循环。"""
    if len(initial_latent) != len(prompt_delta):
        raise ValueError("initial_latent 与 prompt_delta 长度必须一致")
    timesteps = build_timestep_schedule(inference_steps)
    latent = tuple(initial_latent)
    trajectory: list[tuple[float, ...]] = []
    for index, timestep in enumerate(timesteps):
        decay = 1.0 - 0.08 * ((index + 1) / inference_steps)
        guidance = 0.015 * guidance_scale * (0.5 + 0.5 * timestep)
        latent = tuple(decay * value + guidance * delta for value, delta in zip(latent, prompt_delta))
        trajectory.append(latent)
    return SamplingTrace(
        timesteps=timesteps,
        trajectory_vectors=tuple(trajectory),
        metadata={
            "sampler_name": "synthetic_euler_like_sampler",
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
        },
    )
