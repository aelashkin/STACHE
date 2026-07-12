from __future__ import annotations

import copy

import pytest

from stache.envs.minigrid.constants import OBJECT_TO_IDX
from stache.explainability.minigrid.minigrid_neighbor_generation import (
    _direction_neighbors,
    get_neighbors_empty,
    get_neighbors_fetch_old,
)


@pytest.mark.parametrize("direction", range(4))
def test_direction_neighbors_are_exactly_one_left_or_right_turn(direction: int) -> None:
    actual = _direction_neighbors(direction)

    assert actual == ((direction - 1) % 4, (direction + 1) % 4)
    assert len(actual) == len(set(actual)) == 2
    assert (direction + 2) % 4 not in actual


@pytest.mark.parametrize("direction", [-1, 4])
def test_direction_neighbors_reject_out_of_range_values(direction: int) -> None:
    with pytest.raises(ValueError, match=r"direction must be in \[0, 3\]"):
        _direction_neighbors(direction)


@pytest.mark.parametrize("direction", [True, 1.0, "1", None])
def test_direction_neighbors_reject_non_integer_values(direction: object) -> None:
    with pytest.raises(TypeError, match="direction must be an integer"):
        _direction_neighbors(direction)


def _outer_walls(width: int, height: int) -> list[tuple[int, int]]:
    return sorted(
        {(x, 0) for x in range(width)}
        | {(x, height - 1) for x in range(width)}
        | {(0, y) for y in range(height)}
        | {(width - 1, y) for y in range(height)}
    )


def _state(direction: int) -> dict[str, object]:
    return {
        "direction": direction,
        "objects": [[OBJECT_TO_IDX["agent"], 0, 0, 2, 2]],
        "outer_walls": _outer_walls(5, 5),
        "goal": [OBJECT_TO_IDX["key"], 0],
    }


def _direction_only_values(
    state: dict[str, object], neighbors: list[dict[str, object]]
) -> tuple[int, ...]:
    values: list[int] = []
    for neighbor in neighbors:
        candidate = copy.deepcopy(neighbor)
        value = candidate.pop("direction")
        original = copy.deepcopy(state)
        original.pop("direction")
        if candidate == original:
            assert isinstance(value, int)
            values.append(value)
    return tuple(values)


@pytest.mark.parametrize("direction", range(4))
def test_empty_and_fetch_generators_share_turn_adjacency(direction: int) -> None:
    state = _state(direction)
    expected = _direction_neighbors(direction)

    empty_neighbors = get_neighbors_empty(state, env_dimensions=(5, 5))
    fetch_neighbors = get_neighbors_fetch_old(state, max_gen_objects=1)

    assert _direction_only_values(state, empty_neighbors) == expected
    assert _direction_only_values(state, fetch_neighbors) == expected


def test_direction_neighbors_match_minigrid_turn_actions() -> None:
    pytest.importorskip("minigrid", minversion="3.0.0")
    from minigrid.envs import EmptyEnv

    env = EmptyEnv(size=5)
    try:
        env.reset(seed=0)
        for direction in range(4):
            env.agent_dir = direction
            env.step(env.actions.left)
            left = env.agent_dir

            env.agent_dir = direction
            env.step(env.actions.right)
            right = env.agent_dir

            assert _direction_neighbors(direction) == (left, right)
            assert (direction + 2) % 4 not in (left, right)
    finally:
        env.close()
