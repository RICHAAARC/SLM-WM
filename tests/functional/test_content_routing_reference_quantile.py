"""Validate the exact CPU nearest-rank content-reference aggregation kernel."""

from __future__ import annotations

from collections import deque
from dataclasses import FrozenInstanceError, fields
import inspect

import pytest
import torch

import experiments.protocol.content_routing_reference_quantile as quantile_module
from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
    aggregate_content_routing_reference_scalars,
)


pytestmark = pytest.mark.unit


def _member(values: list[float] | tuple[float, ...]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)


def _aggregate_same_population(
    observations: list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> ContentRoutingReferenceScalars:
    return aggregate_content_routing_reference_scalars(
        observations,
        observations,
        observations,
    )


def test_reference_scalars_are_frozen_with_exact_fields() -> None:
    assert tuple(field.name for field in fields(ContentRoutingReferenceScalars)) == (
        "reference_gradient",
        "reference_response",
        "reference_sensitivity",
    )
    result = _aggregate_same_population([_member([1.0])])
    with pytest.raises(FrozenInstanceError):
        result.reference_gradient = 2.0  # type: ignore[misc]


def test_exact_list_and_tuple_inputs_are_equivalent_without_mutation() -> None:
    member_a = _member([1.0, 2.0, 3.0])
    member_b = _member([4.0, 5.0])
    list_observations = [member_a, member_b]
    tuple_observations = (member_a, member_b)
    original_ids = tuple(id(value) for value in list_observations)

    from_list = _aggregate_same_population(list_observations)
    from_tuple = _aggregate_same_population(tuple_observations)

    assert from_list == from_tuple
    assert tuple(id(value) for value in list_observations) == original_ids
    assert len(list_observations) == 2


@pytest.mark.parametrize(
    "invalid_container",
    [
        _member([1.0]),
        (_member([1.0]) for _ in range(1)),
        iter([_member([1.0])]),
        {"member": _member([1.0])},
        "member",
        b"member",
        deque([_member([1.0])]),
    ],
)
def test_non_exact_sequence_containers_are_rejected(
    invalid_container: object,
) -> None:
    valid = [_member([1.0])]
    with pytest.raises(TypeError, match="exact list or tuple"):
        aggregate_content_routing_reference_scalars(
            invalid_container,
            valid,
            valid,
        )


def test_rejected_generator_is_not_consumed() -> None:
    consumed = False

    def observations() -> object:
        nonlocal consumed
        consumed = True
        yield _member([1.0])

    generator = observations()
    valid = [_member([1.0])]
    with pytest.raises(TypeError, match="exact list or tuple"):
        aggregate_content_routing_reference_scalars(generator, valid, valid)
    assert not consumed


@pytest.mark.parametrize("container", [[], ()])
@pytest.mark.parametrize("argument_index", [0, 1, 2])
def test_each_population_container_must_be_nonempty(
    container: object,
    argument_index: int,
) -> None:
    arguments: list[object] = [
        [_member([1.0])],
        [_member([1.0])],
        [_member([1.0])],
    ]
    arguments[argument_index] = container
    with pytest.raises(ValueError, match="at least one observation"):
        aggregate_content_routing_reference_scalars(*arguments)


def test_meta_tensor_fails_device_gate_before_content_access() -> None:
    meta_member = torch.empty((1, 1, 2, 3), dtype=torch.float32, device="meta")
    valid = [_member([1.0])]
    with pytest.raises(ValueError, match="materialized on CPU"):
        aggregate_content_routing_reference_scalars([meta_member], valid, valid)


@pytest.mark.parametrize(
    "invalid_member",
    [
        1.0,
        torch.ones((1, 1, 1, 1), dtype=torch.float64),
        torch.ones((1, 1, 1, 1), dtype=torch.int64),
        torch.ones((1, 1, 1, 1), dtype=torch.bool),
        torch.ones((1, 1, 1, 1), dtype=torch.complex64),
        torch.ones((1,), dtype=torch.float32),
        torch.ones((2, 1, 1, 1), dtype=torch.float32),
        torch.ones((1, 2, 1, 1), dtype=torch.float32),
        torch.empty((1, 1, 0, 1), dtype=torch.float32),
        torch.empty((1, 1, 1, 0), dtype=torch.float32),
    ],
)
def test_invalid_member_type_dtype_or_shape_fails_closed(
    invalid_member: object,
) -> None:
    valid = [_member([1.0])]
    with pytest.raises((TypeError, ValueError)):
        aggregate_content_routing_reference_scalars(
            [invalid_member],
            valid,
            valid,
        )


def test_requires_grad_and_noncontiguous_members_are_read_without_mutation() -> None:
    base = torch.arange(1, 13, dtype=torch.float32).reshape(1, 1, 3, 4)
    member = base.transpose(-2, -1).detach().requires_grad_()
    before = member.detach().clone()
    before_shape = member.shape
    before_stride = member.stride()
    before_requires_grad = member.requires_grad
    before_grad = member.grad

    result = _aggregate_same_population([member])

    assert result.reference_gradient == 12.0
    assert torch.equal(member.detach(), before)
    assert member.shape == before_shape
    assert member.stride() == before_stride
    assert member.requires_grad is before_requires_grad
    assert member.grad is before_grad
    assert not member.is_contiguous()


@pytest.mark.parametrize("invalid_value", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_negative_or_nonfinite_values_fail_closed(invalid_value: float) -> None:
    invalid = _member([1.0, invalid_value, 2.0])
    valid = [_member([1.0])]
    with pytest.raises(ValueError):
        aggregate_content_routing_reference_scalars([invalid], valid, valid)


def test_positive_filter_excludes_positive_and_negative_zero() -> None:
    result = _aggregate_same_population([_member([0.0, -0.0, 1.0, 2.0])])
    assert result == ContentRoutingReferenceScalars(2.0, 2.0, 2.0)


def test_population_with_only_zero_values_fails_closed() -> None:
    zero = [_member([0.0, -0.0])]
    positive = [_member([1.0])]
    with pytest.raises(ValueError, match="no strictly positive observations"):
        aggregate_content_routing_reference_scalars(zero, positive, positive)


@pytest.mark.parametrize("n", [1, 2, 19, 20, 21, 39, 40, 41])
def test_nearest_rank_boundaries_select_exact_sorted_binary32_value(n: int) -> None:
    descending = torch.arange(n, 0, -1, dtype=torch.float32).reshape(1, 1, 1, n)
    result = _aggregate_same_population([descending])
    expected_index = (19 * n + 19) // 20 - 1
    expected_value = torch.arange(1, n + 1, dtype=torch.float32)[expected_index].item()

    for actual in (
        result.reference_gradient,
        result.reference_response,
        result.reference_sensitivity,
    ):
        assert actual == expected_value
        assert torch.tensor(actual, dtype=torch.float32).item() == expected_value


def test_integer_rank_protocol_covers_registered_large_population_without_allocation() -> None:
    n = 562_949_953_420_319
    expected_index = (19 * n + 19) // 20 - 1
    assert expected_index == 534_802_455_749_303


def test_all_members_are_concatenated_before_single_rank_selection() -> None:
    first = torch.arange(1, 20, dtype=torch.float32).reshape(1, 1, 1, 19)
    second = _member([100.0])

    result = _aggregate_same_population([first, second])

    assert result == ContentRoutingReferenceScalars(19.0, 19.0, 19.0)


def test_population_order_duplicates_and_repeated_calls_are_deterministic() -> None:
    first = _member([1.0, 2.0, 2.0, 4.0])
    second = _member([3.0, 4.0, 4.0])

    forward = _aggregate_same_population([first, second])
    repeated = _aggregate_same_population([first, second])
    reversed_members = _aggregate_same_population([second, first])

    assert forward == repeated == reversed_members


def test_three_observation_responsibilities_remain_isolated() -> None:
    result = aggregate_content_routing_reference_scalars(
        [_member([1.0])],
        [_member([2.0])],
        [_member([3.0])],
    )
    assert result == ContentRoutingReferenceScalars(
        reference_gradient=1.0,
        reference_response=2.0,
        reference_sensitivity=3.0,
    )


def test_source_locks_integer_sort_protocol_and_forbids_quantile_substitutes() -> None:
    source = inspect.getsource(quantile_module._nearest_rank_p95)
    compact_source = "".join(source.split())

    assert "math.ceil" not in source
    assert "0.95" not in source
    assert "torch.quantile" not in source
    assert "numpy.quantile" not in source
    assert "torch.sort" in source
    assert "index=(19*n+19)//20-1" in compact_source
