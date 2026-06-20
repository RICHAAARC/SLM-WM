"""把采样过程中捕获到的 latent 转为受治理记录。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from experiments.runtime.diffusion.latent_estimator import latent_statistics, vector_digest


@dataclass(frozen=True)
class LatentTraceRecord:
    """描述单个采样位置的 latent 摘要。"""

    run_id: str
    model_family: str
    model_id: str
    backend_name: str
    trajectory_index: int
    timestep: float
    latent_digest: str
    latent_shape: tuple[int, ...]
    latent_mean: float
    latent_std: float
    latent_min: float
    latent_max: float
    unsupported_reason: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


def build_latent_trace_records(
    run_id: str,
    model_family: str,
    model_id: str,
    backend_name: str,
    timesteps: tuple[float, ...],
    trajectory_vectors: tuple[tuple[float, ...], ...],
    unsupported_reason: str,
) -> tuple[LatentTraceRecord, ...]:
    """将 latent 序列转为 JSONL records。"""
    records: list[LatentTraceRecord] = []
    for index, (timestep, values) in enumerate(zip(timesteps, trajectory_vectors)):
        stats = latent_statistics(values)
        records.append(
            LatentTraceRecord(
                run_id=run_id,
                model_family=model_family,
                model_id=model_id,
                backend_name=backend_name,
                trajectory_index=index,
                timestep=timestep,
                latent_digest=vector_digest(values),
                latent_shape=(len(values),),
                unsupported_reason=unsupported_reason,
                metadata={"trace_source": backend_name},
                **stats,
            )
        )
    return tuple(records)
