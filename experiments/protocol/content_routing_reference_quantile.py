"""Aggregate isolated content observations into nearest-rank references.

This module implements only the deterministic CPU aggregation kernel.  It does
not load, validate, write, or identify a persisted reference registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ContentRoutingReferenceScalars:
    """Hold the three content-routing reference scalars selected by rank."""

    reference_gradient: float
    reference_response: float
    reference_sensitivity: float


def _snapshot_observations(value: Any, *, label: str) -> tuple[torch.Tensor, ...]:
    if type(value) not in {list, tuple}:
        raise TypeError(f"{label} must be an exact list or tuple")
    observations = tuple(value)
    if not observations:
        raise ValueError(f"{label} must contain at least one observation")
    return observations


def _positive_population(
    observations: tuple[torch.Tensor, ...],
    *,
    label: str,
) -> torch.Tensor:
    import torch

    positive_members: list[torch.Tensor] = []
    for member_index, member in enumerate(observations):
        member_label = f"{label}[{member_index}]"
        if not isinstance(member, torch.Tensor):
            raise TypeError(f"{member_label} must be a torch.Tensor")
        if member.dtype != torch.float32:
            raise TypeError(f"{member_label} must have dtype torch.float32")
        if member.device.type != "cpu":
            raise ValueError(f"{member_label} must be materialized on CPU")
        if (
            member.ndim != 4
            or member.shape[0] != 1
            or member.shape[1] != 1
            or member.shape[2] <= 0
            or member.shape[3] <= 0
        ):
            raise ValueError(f"{member_label} must have shape [1, 1, H, W]")

        flat_member = member.detach().contiguous().reshape(-1)
        if not bool(torch.isfinite(flat_member).all().item()):
            raise ValueError(f"{member_label} must contain only finite values")
        if bool((flat_member < 0.0).any().item()):
            raise ValueError(f"{member_label} must contain only nonnegative values")
        positive_member = flat_member[flat_member > 0.0]
        if positive_member.numel() > 0:
            positive_members.append(positive_member)

    if not positive_members:
        raise ValueError(f"{label} has no strictly positive observations")
    return torch.cat(positive_members)


def _nearest_rank_p95(positive_values: torch.Tensor) -> float:
    import torch

    n = int(positive_values.numel())
    if n == 0:
        raise ValueError("nearest-rank population must not be empty")
    sorted_values = torch.sort(positive_values).values
    index = (19 * n + 19) // 20 - 1
    return sorted_values[index].item()


def aggregate_content_routing_reference_scalars(
    gradient_observations: Any,
    response_observations: Any,
    sensitivity_observations: Any,
) -> ContentRoutingReferenceScalars:
    """Select exact nearest-rank P95 references from three raw populations."""

    gradient_snapshot = _snapshot_observations(
        gradient_observations,
        label="gradient_observations",
    )
    response_snapshot = _snapshot_observations(
        response_observations,
        label="response_observations",
    )
    sensitivity_snapshot = _snapshot_observations(
        sensitivity_observations,
        label="sensitivity_observations",
    )

    gradient_population = _positive_population(
        gradient_snapshot,
        label="gradient_observations",
    )
    response_population = _positive_population(
        response_snapshot,
        label="response_observations",
    )
    sensitivity_population = _positive_population(
        sensitivity_snapshot,
        label="sensitivity_observations",
    )
    return ContentRoutingReferenceScalars(
        reference_gradient=_nearest_rank_p95(gradient_population),
        reference_response=_nearest_rank_p95(response_population),
        reference_sensitivity=_nearest_rank_p95(sensitivity_population),
    )
