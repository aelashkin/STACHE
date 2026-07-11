"""Domain-neutral contracts for robustness-region connectors.

This module deliberately contains no environment, model, rendering, or
persistence imports.  A connector owns state identity and graph/metric truth;
the generic search only consumes the declarations exposed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Generic, Hashable, Iterable, Protocol, TypeVar, runtime_checkable


StateT = TypeVar("StateT")
StateKeyT = TypeVar("StateKeyT", bound=Hashable)


@dataclass(frozen=True, slots=True)
class ConnectorIdentity:
    """Versioned identity for a connector and its scientific state space."""

    domain: str
    connector_version: str
    state_universe: str
    state_universe_version: str
    metric: str
    metric_version: str
    codec: str = "none"
    codec_version: str = "none"


@dataclass(frozen=True, slots=True)
class MetricCertificate:
    """Connector-owned claims relating its atomic graph to formal distance.

    ``geodesic_for_formal_metric`` is intentionally explicit.  Merely having
    unit, symmetric edges does not prove that a graph-boundary minimum is a
    global formal-distance minimum.
    """

    formal_unit: float
    every_edge_is_formal_unit: bool
    all_valid_formal_unit_edges_present: bool
    symmetric: bool
    connected: bool
    geodesic_for_formal_metric: bool
    certificate_version: str
    scope_fingerprint: str

    @property
    def certifies_global_minimum_from_graph(self) -> bool:
        """Whether graph layers may certify a global formal minimum."""

        return (
            self.every_edge_is_formal_unit
            and self.all_valid_formal_unit_edges_present
            and self.symmetric
            and self.connected
            and self.geodesic_for_formal_metric
        )


@dataclass(frozen=True, slots=True)
class FormalDistanceLayer(Generic[StateT]):
    """A complete connector-provided layer at one formal distance."""

    distance: float
    states: tuple[StateT, ...]


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Minimal model-observation contract, independent of Gymnasium."""

    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class DiscreteActionSpec:
    """Minimal scalar discrete-action contract."""

    count: int


@runtime_checkable
class SearchConnector(Protocol[StateT, StateKeyT]):
    """State identity, declared universe, graph, and metric operations."""

    identity: ConnectorIdentity
    metric_certificate: MetricCertificate
    action_spec: DiscreteActionSpec

    def canonicalize(self, state: StateT) -> StateT:
        """Return the connector's canonical representation for ``state``."""

    def validate_state(self, state: StateT) -> None:
        """Raise a useful exception when ``state`` is outside the universe."""

    def state_key(self, state: StateT) -> StateKeyT:
        """Return the unique, stable identity key for canonical ``state``."""

    def ordering_key(self, key: StateKeyT) -> object:
        """Return a deterministic total-order key for a state key."""

    def declared_states(self) -> Iterable[StateT]:
        """Iterate the complete declared state universe."""

    def atomic_neighbors(self, state: StateT) -> Iterable[StateT]:
        """Iterate states one declared atomic perturbation from ``state``."""

    def formal_distance(self, left: StateT, right: StateT) -> float:
        """Return the normative formal distance between two valid states."""

    def formal_layers(
        self,
        seed: StateT,
    ) -> Iterable[FormalDistanceLayer[StateT]] | None:
        """Optionally provide complete increasing formal-distance layers."""


@runtime_checkable
class PolicyConnector(Protocol[StateT, StateKeyT]):
    """Connector surface needed by model and table action sources."""

    observation_spec: ObservationSpec
    action_spec: DiscreteActionSpec

    def canonicalize(self, state: StateT) -> StateT:
        """Return the connector's canonical representation for ``state``."""

    def validate_state(self, state: StateT) -> None:
        """Raise when ``state`` is not in the connector's universe."""

    def state_key(self, state: StateT) -> StateKeyT:
        """Return the stable identity used by the shared action cache."""

    def encode_observation(self, state: StateT) -> object:
        """Encode a canonical state for the policy model."""

    def policy_lookup_key(self, state: StateT) -> Hashable:
        """Return the key used by a precomputed policy table."""


@dataclass(frozen=True, slots=True)
class ExactActionInvariance:
    """Exact equality over already-normalized scalar discrete actions."""

    fingerprint: ClassVar[str] = "exact-action-equality-v1"

    def __call__(self, seed_action: int, candidate_action: int) -> bool:
        return seed_action == candidate_action

    def equivalent(self, seed_action: int, candidate_action: int) -> bool:
        """Named spelling for callers that prefer an explicit predicate."""

        return self(seed_action, candidate_action)
