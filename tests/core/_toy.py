"""Declarative toy state spaces and an RR/CF oracle independent of production search."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from stache.explainability.core.connector import (
    ConnectorIdentity,
    DiscreteActionSpec,
    FormalDistanceLayer,
    MetricCertificate,
)
from stache.explainability.core.policy import OracleStats, custom_policy_source


Edge = tuple[str, str]


def normalized_edge(left: str, right: str) -> Edge:
    if left == right:
        raise ValueError("toy edges may not be self-loops")
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def unit_edges(coordinates: Mapping[str, tuple[int, ...]]) -> frozenset[Edge]:
    """Return every pair in the declared universe at Manhattan distance one."""

    states = tuple(coordinates)
    return frozenset(
        normalized_edge(left, right)
        for index, left in enumerate(states)
        for right in states[index + 1 :]
        if manhattan(coordinates[left], coordinates[right]) == 1
    )


def manhattan(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("toy factor vectors must have equal dimensions")
    return float(sum(abs(a - b) for a, b in zip(left, right, strict=True)))


@dataclass(frozen=True)
class ToySpace:
    name: str
    states: tuple[str, ...]
    edges: frozenset[Edge]
    coordinates: Mapping[str, tuple[int, ...]]
    actions: Mapping[str, int]
    connected: bool = True
    geodesic: bool = True
    all_unit_edges_present: bool = True
    provide_formal_layers: bool = False

    def __post_init__(self) -> None:
        state_set = set(self.states)
        if state_set != set(self.coordinates) or state_set != set(self.actions):
            raise ValueError("states, coordinates, and actions must have identical keys")
        if any(set(edge) - state_set for edge in self.edges):
            raise ValueError("every edge endpoint must be a declared state")
        if any(
            manhattan(self.coordinates[left], self.coordinates[right]) != 1.0
            for left, right in self.edges
        ):
            raise ValueError("every toy edge must be a formal unit edge")


class ToyConnector:
    """Small strict connector whose graph can be iterated in either input order."""

    def __init__(self, space: ToySpace, *, reverse_neighbors: bool = False) -> None:
        self.space = space
        self.reverse_neighbors = reverse_neighbors
        self.action_spec = DiscreteActionSpec(count=max(space.actions.values()) + 1)
        self.identity = ConnectorIdentity(
            domain="toy",
            connector_version="1",
            state_universe=space.name,
            state_universe_version="1",
            metric="toy-manhattan",
            metric_version="1",
            object_projection="toy-canonical-string",
            object_projection_version="1",
            factorization="toy-coordinate-vector",
            factorization_version="1",
            topology="formal-unit-edges",
            topology_version="1",
            adjacency_threshold=1.0,
        )
        self.metric_certificate = MetricCertificate(
            formal_unit=1.0,
            every_edge_is_formal_unit=True,
            all_valid_formal_unit_edges_present=space.all_unit_edges_present,
            symmetric=True,
            connected=space.connected,
            geodesic_for_formal_metric=space.geodesic,
            certificate_version="toy-certificate-v1",
            scope_fingerprint=f"toy-scope:{space.name}",
        )

    def canonicalize(self, state: str) -> str:
        if not isinstance(state, str):
            raise TypeError(f"toy state must be str, got {type(state).__name__}")
        return state.strip().lower()

    def validate_state(self, state: str) -> None:
        if state not in self.space.states:
            raise ValueError(f"unknown toy state: {state!r}")

    def state_key(self, state: str) -> str:
        return state

    def policy_lookup_key(self, state: str) -> str:
        return state

    def ordering_key(self, key: str) -> tuple[str]:
        return (key,)

    def declared_states(self) -> Iterable[str]:
        return reversed(self.space.states) if self.reverse_neighbors else self.space.states

    def atomic_neighbors(self, state: str) -> Iterable[str]:
        neighbors = [
            right if left == state else left
            for left, right in self.space.edges
            if state == left or state == right
        ]
        neighbors.sort(reverse=self.reverse_neighbors)
        return tuple(neighbors)

    def formal_distance(self, left: str, right: str) -> float:
        return manhattan(self.space.coordinates[left], self.space.coordinates[right])

    def formal_layers(self, seed: str) -> Iterable[FormalDistanceLayer[str]] | None:
        if not self.space.provide_formal_layers:
            return None
        grouped: dict[float, list[str]] = defaultdict(list)
        for state in self.space.states:
            grouped[self.formal_distance(seed, state)].append(state)
        return tuple(
            FormalDistanceLayer(distance=distance, states=tuple(sorted(states)))
            for distance, states in sorted(grouped.items())
        )


class ToyOracle:
    """Counting exact-action oracle with a checkpointable canonical-state cache."""

    def __init__(
        self,
        actions: Mapping[str, int],
        *,
        fingerprint: str = "toy-policy-v1",
    ) -> None:
        self.actions = dict(actions)
        self.fingerprint, self.source_description = custom_policy_source(
            {
                "provider": "tests.core.ToyOracle",
                "declared_fingerprint": fingerprint,
                "actions": [
                    [state, action]
                    for state, action in sorted(self.actions.items())
                ],
            }
        )
        self.calls: list[str] = []
        self._cache: dict[str, int] = {}
        self._policy_queries = 0
        self._cache_hits = 0

    def action(self, state: str) -> int:
        if state in self._cache:
            self._cache_hits += 1
            return self._cache[state]
        action = self.actions[state]
        self._cache[state] = action
        self._policy_queries += 1
        self.calls.append(state)
        return action

    def policy_query_cost(self, state: str) -> int:
        return 0 if state in self._cache else 1

    def has_cached(self, key: str) -> bool:
        return key in self._cache

    @property
    def stats(self) -> OracleStats:
        return OracleStats(
            policy_queries=self._policy_queries,
            cache_hits=self._cache_hits,
            table_hits=self._policy_queries,
        )

    def export_cache(self) -> Mapping[str, int]:
        return dict(self._cache)

    def restore_cache(self, cache: Mapping[str, int]) -> None:
        self._cache = dict(cache)


@dataclass(frozen=True)
class BruteForceResult:
    region: frozenset[str]
    boundary: frozenset[str]
    graph_depths: Mapping[str, int]
    graph_minimal: frozenset[str]
    graph_radius: float | None
    formal_minimal: frozenset[str]
    formal_radius: float | None


def brute_force(space: ToySpace, seed: str) -> BruteForceResult:
    """Compute expected science without calling a connector or production search.

    RR membership is a fixed-point closure, boundary membership is a full edge scan,
    graph depth is a separate layer-set traversal, and formal minima scan every
    declared policy-changing state directly.
    """

    adjacency = {state: set() for state in space.states}
    for left, right in space.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    seed_action = space.actions[seed]
    region = {seed}
    while True:
        additions = {
            state
            for state in space.states
            if state not in region
            and space.actions[state] == seed_action
            and adjacency[state].intersection(region)
        }
        if not additions:
            break
        region.update(additions)

    boundary = {
        state
        for state in space.states
        if space.actions[state] != seed_action
        and adjacency[state].intersection(region)
    }

    depths = {seed: 0}
    visited = {seed}
    invariant_frontier = {seed}
    depth = 0
    while invariant_frontier:
        candidates = set().union(*(adjacency[state] for state in invariant_frontier))
        candidates.difference_update(visited)
        if not candidates:
            break
        depth += 1
        for state in candidates:
            depths[state] = depth
        visited.update(candidates)
        invariant_frontier = {
            state for state in candidates if space.actions[state] == seed_action
        }

    if boundary:
        minimum_graph_depth = min(depths[state] for state in boundary)
        graph_minimal = {
            state for state in boundary if depths[state] == minimum_graph_depth
        }
        graph_radius: float | None = float(minimum_graph_depth)
    else:
        graph_minimal = set()
        graph_radius = None

    global_counterfactuals = {
        state for state in space.states if space.actions[state] != seed_action
    }
    if global_counterfactuals:
        formal_radius = min(
            manhattan(space.coordinates[seed], space.coordinates[state])
            for state in global_counterfactuals
        )
        formal_minimal = {
            state
            for state in global_counterfactuals
            if manhattan(space.coordinates[seed], space.coordinates[state])
            == formal_radius
        }
    else:
        formal_radius = None
        formal_minimal = set()

    return BruteForceResult(
        region=frozenset(region),
        boundary=frozenset(boundary),
        graph_depths=depths,
        graph_minimal=frozenset(graph_minimal),
        graph_radius=graph_radius,
        formal_minimal=frozenset(formal_minimal),
        formal_radius=formal_radius,
    )


def exact_space() -> ToySpace:
    coordinates = {
        "s": (0, 0),
        "a": (1, 0),
        "b": (0, 1),
        "c": (1, 1),
        "d": (1, 2),
        "e": (2, 0),
        "f": (2, 1),
        "g": (3, 1),
    }
    return ToySpace(
        name="exact-diamond",
        states=tuple(coordinates),
        edges=unit_edges(coordinates),
        coordinates=coordinates,
        actions={"s": 0, "a": 0, "b": 0, "c": 0, "d": 2, "e": 1, "f": 0, "g": 1},
    )


def tied_minimum_space() -> ToySpace:
    coordinates = {
        "s": (0, 0),
        "a": (-1, 0),
        "b": (1, 0),
        "c": (0, 1),
        "x": (-2, 0),
        "y": (2, 0),
        "z": (0, 2),
        "w": (0, 3),
    }
    return ToySpace(
        name="tied-minimum",
        states=tuple(coordinates),
        edges=unit_edges(coordinates),
        coordinates=coordinates,
        actions={"s": 0, "a": 0, "b": 0, "c": 0, "x": 1, "y": 2, "z": 0, "w": 3},
    )


def query_budget_space() -> ToySpace:
    coordinates = {"s": (0, 0), "a": (-1, 0), "b": (1, 0), "c": (0, 1)}
    return ToySpace(
        name="query-budget-ties",
        states=tuple(coordinates),
        edges=unit_edges(coordinates),
        coordinates=coordinates,
        actions={"s": 0, "a": 1, "b": 2, "c": 3},
    )


def no_counterfactual_space() -> ToySpace:
    coordinates = {"s": (0,), "a": (1,), "b": (2,)}
    return ToySpace(
        name="constant-policy-line",
        states=tuple(coordinates),
        edges=unit_edges(coordinates),
        coordinates=coordinates,
        actions={state: 0 for state in coordinates},
    )


def disconnected_unknown_space() -> ToySpace:
    coordinates = {"s": (0,), "a": (1,), "i": (10,), "q": (11,)}
    return ToySpace(
        name="disconnected-global-cf",
        states=tuple(coordinates),
        edges=frozenset({normalized_edge("s", "a"), normalized_edge("i", "q")}),
        coordinates=coordinates,
        actions={"s": 0, "a": 0, "i": 0, "q": 1},
        connected=False,
        geodesic=False,
        all_unit_edges_present=True,
    )


def non_geodesic_space(*, provide_formal_layers: bool = True) -> ToySpace:
    coordinates = {
        "s": (0, 0),
        "a": (-1, 0),
        "b": (-2, 0),
        "g": (-3, 0),
        "c": (0, 1),
        "d": (1, 1),
        "e": (2, 1),
        "h": (2, 0),
    }
    edges = frozenset(
        normalized_edge(*edge)
        for edge in (
            ("s", "a"),
            ("a", "b"),
            ("b", "g"),
            ("s", "c"),
            ("c", "d"),
            ("d", "e"),
            ("e", "h"),
        )
    )
    return ToySpace(
        name=f"non-geodesic-layers-{provide_formal_layers}",
        states=tuple(coordinates),
        edges=edges,
        coordinates=coordinates,
        actions={"s": 0, "a": 0, "b": 0, "g": 1, "c": 0, "d": 0, "e": 0, "h": 2},
        geodesic=False,
        provide_formal_layers=provide_formal_layers,
    )


def disconnected_formal_minimum_space() -> ToySpace:
    coordinates = {"s": (0, 0), "a": (-1, 0), "g": (-2, 0), "q": (0, 1)}
    return ToySpace(
        name="disconnected-formal-minimum",
        states=tuple(coordinates),
        edges=frozenset({normalized_edge("s", "a"), normalized_edge("a", "g")}),
        coordinates=coordinates,
        actions={"s": 0, "a": 0, "g": 1, "q": 2},
        connected=False,
        geodesic=False,
        all_unit_edges_present=False,
        provide_formal_layers=True,
    )
