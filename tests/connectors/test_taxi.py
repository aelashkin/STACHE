"""Exhaustive contracts for the thesis-compatible Taxi connector.

Expected values are derived here from the four-factor Taxi state space.  The
tests intentionally do not call Gymnasium's Taxi encoder or any legacy STACHE
Taxi helper, so they can detect drift in the connector's universe, metric, and
perturbation relation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np
import pytest

from stache.explainability.connectors.taxi import TaxiConnector
from stache.explainability.core.models import (
    CounterfactualExistence,
    CounterfactualSelection,
    MinimumBasis,
    SearchExtent,
    SearchOptions,
)
from stache.explainability.core.policy import TableActionOracle
from stache.explainability.core.search import compute_rr


TaxiTuple = tuple[int, int, int, int]
ALL_STATES: tuple[TaxiTuple, ...] = tuple(
    product(range(5), range(5), range(5), range(4))
)
ALL_STATE_SET = frozenset(ALL_STATES)


def _independent_index(state: TaxiTuple) -> int:
    row, column, passenger, destination = state
    return (((row * 5) + column) * 5 + passenger) * 4 + destination


def _independent_distance(left: TaxiTuple, right: TaxiTuple) -> int:
    return (
        abs(left[0] - right[0])
        + abs(left[1] - right[1])
        + int(left[2] != right[2])
        + int(left[3] != right[3])
    )


def _independent_neighbors(state: TaxiTuple) -> tuple[TaxiTuple, ...]:
    row, column, passenger, destination = state
    neighbors: list[TaxiTuple] = []

    for candidate_row in (row - 1, row + 1):
        if 0 <= candidate_row < 5:
            neighbors.append((candidate_row, column, passenger, destination))
    for candidate_column in (column - 1, column + 1):
        if 0 <= candidate_column < 5:
            neighbors.append((row, candidate_column, passenger, destination))
    neighbors.extend(
        (row, column, candidate_passenger, destination)
        for candidate_passenger in range(5)
        if candidate_passenger != passenger
    )
    neighbors.extend(
        (row, column, passenger, candidate_destination)
        for candidate_destination in range(4)
        if candidate_destination != destination
    )
    return tuple(neighbors)


def _assert_primitive(value: Any) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_primitive(item)
        return
    if isinstance(value, Mapping):
        assert all(isinstance(key, str) for key in value)
        for item in value.values():
            _assert_primitive(item)
        return
    pytest.fail(f"non-primitive value {value!r} ({type(value).__name__})")


def _independent_rr_and_boundary(
    seed: TaxiTuple,
    table: Mapping[int, int],
) -> tuple[
    frozenset[TaxiTuple],
    frozenset[TaxiTuple],
    Mapping[TaxiTuple, int],
]:
    """Return the graph fixed point without calling production connector/search."""

    seed_action = table[_independent_index(seed)]
    region = {seed}
    boundary: set[TaxiTuple] = set()
    discovered_depth = {seed: 0}
    queue: deque[TaxiTuple] = deque([seed])

    while queue:
        state = queue.popleft()
        for neighbor in _independent_neighbors(state):
            if neighbor in discovered_depth:
                continue
            discovered_depth[neighbor] = discovered_depth[state] + 1
            if table[_independent_index(neighbor)] == seed_action:
                region.add(neighbor)
                queue.append(neighbor)
            else:
                boundary.add(neighbor)

    return frozenset(region), frozenset(boundary), discovered_depth


@pytest.fixture(scope="module")
def connector() -> TaxiConnector:
    return TaxiConnector()


def test_declares_exact_thesis_500_universe_in_deterministic_order(
    connector: TaxiConnector,
) -> None:
    declared = tuple(connector.declared_states())

    assert declared == ALL_STATES
    assert len(declared) == 500
    assert len(set(declared)) == 500
    assert (
        sum(passenger == destination for _, _, passenger, destination in declared)
        == 100
    )
    assert (0, 0, 0, 0) in declared
    assert (4, 4, 3, 3) in declared


def test_connector_identity_and_policy_space_are_stable(
    connector: TaxiConnector,
) -> None:
    identity = connector.identity

    assert identity.domain == "taxi"
    assert identity.connector_version == "1"
    assert identity.state_universe == "taxi-factored-500"
    assert identity.state_universe_version == "1"
    assert identity.metric == "taxi-thesis-hybrid"
    assert identity.metric_version == "1"
    assert identity.codec == "taxi-state-key"
    assert identity.codec_version == "1"
    assert connector.observation_codec_version == "1"
    assert connector.observation_spec.shape == (500,)
    assert connector.observation_spec.dtype == "float32"
    assert connector.action_spec.count == 6
    assert connector.artifact_codec is connector


def test_metric_certificate_truthfully_covers_the_declared_universe(
    connector: TaxiConnector,
) -> None:
    certificate = connector.metric_certificate

    assert certificate.formal_unit == 1
    assert certificate.every_edge_is_formal_unit is True
    assert certificate.all_valid_formal_unit_edges_present is True
    assert certificate.symmetric is True
    assert certificate.connected is True
    assert certificate.geodesic_for_formal_metric is True
    assert certificate.certificate_version == "1"
    assert isinstance(certificate.scope_fingerprint, str)
    assert certificate.scope_fingerprint
    assert connector.formal_layers((0, 0, 0, 0)) is None


@pytest.mark.parametrize(
    ("invalid_state", "message_fragment"),
    [
        ((-1, 0, 0, 0), "row"),
        ((5, 0, 0, 0), "row"),
        ((0, -1, 0, 0), "column"),
        ((0, 5, 0, 0), "column"),
        ((0, 0, -1, 0), "passenger"),
        ((0, 0, 5, 0), "passenger"),
        ((0, 0, 0, -1), "destination"),
        ((0, 0, 0, 4), "destination"),
        ((0, 0, 0), "four"),
        ((0, 0, 0, 0, 0), "four"),
        ((0.0, 0, 0, 0), "integer"),
        ((False, 0, 0, 0), "integer"),
    ],
)
def test_invalid_states_are_rejected_with_factor_diagnostics(
    connector: TaxiConnector,
    invalid_state: tuple[object, ...],
    message_fragment: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message_fragment):
        connector.validate_state(invalid_state)  # type: ignore[arg-type]


def test_canonical_identity_and_ordering_are_total_over_all_states(
    connector: TaxiConnector,
) -> None:
    keys: set[TaxiTuple] = set()
    ordering_keys: set[int] = set()

    for expected_index, state in enumerate(ALL_STATES):
        canonical = connector.canonicalize(state)
        connector.validate_state(canonical)
        key = connector.state_key(canonical)
        ordering_key = connector.ordering_key(key)

        assert canonical == state
        assert isinstance(canonical, tuple)
        assert key == state
        assert ordering_key == expected_index
        assert connector.canonicalize(canonical) == canonical
        keys.add(key)
        ordering_keys.add(ordering_key)

    assert keys == ALL_STATE_SET
    assert ordering_keys == set(range(500))


def test_index_policy_key_and_primitive_codecs_round_trip_all_states(
    connector: TaxiConnector,
) -> None:
    for expected_index, state in enumerate(ALL_STATES):
        assert _independent_index(state) == expected_index
        assert connector.encode_index(state) == expected_index
        assert connector.policy_lookup_key(state) == expected_index
        assert connector.decode_index(expected_index) == state

        encoded_state = connector.encode_state(state)
        encoded_key = connector.encode_key(connector.state_key(state))
        assert encoded_state == list(state)
        assert encoded_key == list(state)
        _assert_primitive(encoded_state)
        _assert_primitive(encoded_key)
        assert connector.decode_state(encoded_state) == state
        assert connector.decode_key(encoded_key) == state


@pytest.mark.parametrize("index", [-1, 500, True, 1.0, "1"])
def test_decode_index_rejects_out_of_range_or_lossy_values(
    connector: TaxiConnector,
    index: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="index"):
        connector.decode_index(index)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "encoded",
    [
        [0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 4],
        [0, 0, 0, 0.0],
        [0, 0, 0, False],
        {"row": 0, "column": 0, "passenger": 0, "destination": 0},
        "[0, 0, 0, 0]",
    ],
)
def test_artifact_decoders_reject_malformed_or_lossy_values(
    connector: TaxiConnector,
    encoded: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        connector.decode_state(encoded)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        connector.decode_key(encoded)  # type: ignore[arg-type]


def test_one_hot_observation_round_trip_for_every_index(
    connector: TaxiConnector,
) -> None:
    observations: set[bytes] = set()

    for expected_index, state in enumerate(ALL_STATES):
        observation = connector.encode_observation(state)

        assert isinstance(observation, np.ndarray)
        assert observation.shape == (500,)
        assert observation.dtype == np.float32
        assert np.count_nonzero(observation) == 1
        assert observation.sum() == np.float32(1.0)
        assert observation[expected_index] == np.float32(1.0)
        assert np.all(observation[:expected_index] == 0)
        assert np.all(observation[expected_index + 1 :] == 0)
        observations.add(observation.tobytes())

    assert len(observations) == 500


def test_exact_neighbor_relation_is_exhaustive_valid_unique_and_symmetric(
    connector: TaxiConnector,
) -> None:
    actual_by_state: dict[TaxiTuple, tuple[TaxiTuple, ...]] = {}

    for state in ALL_STATES:
        actual = tuple(connector.atomic_neighbors(state))
        expected = _independent_neighbors(state)
        actual_by_state[state] = actual

        assert actual == expected
        assert len(actual) == len(set(actual))
        assert state not in actual
        assert set(actual) <= ALL_STATE_SET
        assert all(_independent_distance(state, neighbor) == 1 for neighbor in actual)
        assert all(connector.formal_distance(state, neighbor) == 1 for neighbor in actual)

    for state, neighbors in actual_by_state.items():
        for neighbor in neighbors:
            assert state in actual_by_state[neighbor]


def test_road_walls_do_not_restrict_thesis_perturbation_edges(
    connector: TaxiConnector,
) -> None:
    # Taxi's MDP map has an internal wall between these cells.  The thesis
    # perturbation graph changes the column factor directly and must include it.
    left_of_wall = (0, 1, 0, 0)
    right_of_wall = (0, 2, 0, 0)

    assert right_of_wall in connector.atomic_neighbors(left_of_wall)
    assert left_of_wall in connector.atomic_neighbors(right_of_wall)


def test_graph_distance_equals_formal_distance_for_all_250_000_pairs(
    connector: TaxiConnector,
) -> None:
    # Adjacency is built from the independent factor definition, not from the
    # connector under test.  Running one BFS per source proves the certificate's
    # all-pairs geodesy and connectivity claims over the complete finite universe.
    independent_adjacency = {
        state: _independent_neighbors(state) for state in ALL_STATES
    }

    compared_pairs = 0
    for source in ALL_STATES:
        graph_depth: dict[TaxiTuple, int] = {source: 0}
        queue: deque[TaxiTuple] = deque([source])
        while queue:
            state = queue.popleft()
            for neighbor in independent_adjacency[state]:
                if neighbor not in graph_depth:
                    graph_depth[neighbor] = graph_depth[state] + 1
                    queue.append(neighbor)

        assert len(graph_depth) == 500
        for target in ALL_STATES:
            expected = _independent_distance(source, target)
            assert graph_depth[target] == expected
            assert connector.formal_distance(source, target) == expected
            compared_pairs += 1

    assert compared_pairs == 250_000


def test_action_metadata_is_rendering_independent_and_primitive(
    connector: TaxiConnector,
) -> None:
    metadata: Sequence[Mapping[str, object]] = connector.action_metadata

    assert tuple(item["action"] for item in metadata) == tuple(range(6))
    assert tuple(str(item["label"]).lower() for item in metadata) == (
        "south",
        "north",
        "east",
        "west",
        "pickup",
        "dropoff",
    )
    _assert_primitive(metadata)


@pytest.mark.parametrize(
    "basis",
    [MinimumBasis.GRAPH_BOUNDARY, MinimumBasis.FORMAL_GLOBAL],
)
def test_arbitrary_500_state_table_matches_independent_rr_cf_oracle(
    connector: TaxiConnector,
    basis: MinimumBasis,
) -> None:
    # Deliberately irregular and unrelated to Taxi dynamics.  Exhaustive table
    # lookup makes policy behavior fully specified for all thesis states.
    table = {
        index: (index * 37 + index // 11 + (index % 7) * 3) % 6
        for index in range(500)
    }
    seed = (2, 2, 4, 1)
    seed_action = table[_independent_index(seed)]
    expected_region, expected_boundary, graph_depths = _independent_rr_and_boundary(
        seed,
        table,
    )
    expected_formal_candidates = {
        state
        for state in ALL_STATES
        if table[_independent_index(state)] != seed_action
    }

    if basis is MinimumBasis.GRAPH_BOUNDARY:
        radius = min(graph_depths[state] for state in expected_boundary)
        expected_minima = {
            state for state in expected_boundary if graph_depths[state] == radius
        }
    else:
        radius = min(
            _independent_distance(seed, state)
            for state in expected_formal_candidates
        )
        expected_minima = {
            state
            for state in expected_formal_candidates
            if _independent_distance(seed, state) == radius
        }

    result = compute_rr(
        seed,
        connector,
        TableActionOracle(connector, table, source_fingerprint="taxi-table-test-v1"),
        SearchOptions(
            counterfactuals=CounterfactualSelection.BOTH,
            minimum_basis=basis,
            extent=SearchExtent.EXACT,
        ),
    )

    assert {record.state for record in result.region} == expected_region
    assert {
        record.state for record in result.boundary_counterfactuals
    } == expected_boundary
    assert {record.state for record in result.minimal_counterfactuals} == expected_minima
    assert result.seed.action == seed_action
    assert result.robustness_radius == float(radius)
    assert result.counterfactual_existence is CounterfactualExistence.FOUND
    assert result.completeness.region_complete is True
    assert result.completeness.boundary_complete is True
    assert result.completeness.radius_complete is True
    assert result.completeness.minimal_counterfactuals_complete is True
