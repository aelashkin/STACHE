"""Immutable public models for domain-neutral robustness-region search.

The objects in this module describe scientific meaning separately from runtime
resource ceilings.  Persistence is intentionally implemented outside the core.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Mapping, TypeVar

from .connector import ConnectorIdentity, MetricCertificate


StateT = TypeVar("StateT")
KeyT = TypeVar("KeyT")
CORE_SCHEMA_VERSION = 2


class InvalidSearchOptions(ValueError):
    """Raised when search options would create an ambiguous result contract."""


class MetricCertificationError(ValueError):
    """Raised when a requested formal claim lacks a sufficient certificate."""


class ContinuationMismatchError(ValueError):
    """Raised when a continuation does not belong to the requested search."""


class SearchInvariantError(RuntimeError):
    """Raised when a connector violates identity or graph invariants."""


class CounterfactualSelection(str, Enum):
    MINIMAL = "minimal"
    BOUNDARY = "boundary"
    BOTH = "both"


class MinimumBasis(str, Enum):
    GRAPH_BOUNDARY = "graph_boundary"
    FORMAL_GLOBAL = "formal_global"


class SearchExtent(str, Enum):
    EXACT = "exact"
    THROUGH_MINIMAL_CF = "through_minimal_cf"


class CounterfactualExistence(str, Enum):
    FOUND = "found"
    PROVEN_ABSENT = "proven_absent"
    UNKNOWN = "unknown"


class StopReason(str, Enum):
    COMPLETE = "complete"
    THROUGH_MINIMAL = "through_minimal"
    MAX_EXPANDED = "max_expanded"
    MAX_POLICY_QUERIES = "max_policy_queries"
    MAX_GRAPH_DEPTH = "max_graph_depth"


def _coerce_enum(value: Any, enum_type: type[Enum], field: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise InvalidSearchOptions(f"{field} must be one of {choices}") from exc


def _validate_ceiling(value: int | None, field: str, *, minimum: int) -> None:
    if value is None:
        return
    if type(value) is not int or value < minimum:
        raise InvalidSearchOptions(
            f"{field} must be an integer >= {minimum} or None"
        )


@dataclass(frozen=True)
class SearchOptions:
    """Scientific options plus independent total resource ceilings."""

    counterfactuals: CounterfactualSelection = CounterfactualSelection.BOTH
    minimum_basis: MinimumBasis = MinimumBasis.GRAPH_BOUNDARY
    extent: SearchExtent = SearchExtent.EXACT
    max_expanded: int | None = None
    max_policy_queries: int | None = None
    max_graph_depth: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "counterfactuals",
            _coerce_enum(
                self.counterfactuals,
                CounterfactualSelection,
                "counterfactuals",
            ),
        )
        object.__setattr__(
            self,
            "minimum_basis",
            _coerce_enum(self.minimum_basis, MinimumBasis, "minimum_basis"),
        )
        object.__setattr__(
            self,
            "extent",
            _coerce_enum(self.extent, SearchExtent, "extent"),
        )
        _validate_ceiling(self.max_expanded, "max_expanded", minimum=0)
        _validate_ceiling(
            self.max_policy_queries,
            "max_policy_queries",
            minimum=0,
        )
        _validate_ceiling(self.max_graph_depth, "max_graph_depth", minimum=0)

        if (
            self.extent is SearchExtent.THROUGH_MINIMAL_CF
            and self.counterfactuals is not CounterfactualSelection.MINIMAL
        ):
            raise InvalidSearchOptions(
                "through_minimal_cf can only project minimal counterfactuals"
            )

    def semantic_values(self) -> Mapping[str, str]:
        """Return fields that must remain fixed when a search is resumed."""

        return {
            "counterfactuals": self.counterfactuals.value,
            "minimum_basis": self.minimum_basis.value,
            "extent": self.extent.value,
        }


@dataclass(frozen=True)
class StateRecord(Generic[StateT, KeyT]):
    state: StateT
    key: KeyT
    action: int
    graph_depth: int | None
    formal_distance: int | float
    discovery_source: str = "graph"


@dataclass(frozen=True)
class CounterfactualProjection(Generic[StateT, KeyT]):
    """Explicit output projection; ``None`` means omitted, not empty."""

    minimal: tuple[StateRecord[StateT, KeyT], ...] | None
    boundary: tuple[StateRecord[StateT, KeyT], ...] | None


@dataclass(frozen=True)
class SearchCompleteness:
    region_complete: bool
    boundary_complete: bool
    radius_complete: bool
    minimal_counterfactuals_complete: bool
    max_evaluated_graph_depth: int
    max_expanded_graph_depth: int | None
    max_scanned_formal_distance: int | float | None
    remaining_frontier_size: int
    stop_reason: StopReason


@dataclass(frozen=True)
class SearchStats:
    states_discovered: int
    states_evaluated: int
    states_expanded: int
    policy_queries: int
    cache_hits: int = 0
    table_hits: int = 0
    model_queries: int = 0
    duplicate_discoveries: int = 0
    formal_states_scanned: int = 0
    resume_count: int = 0


@dataclass(frozen=True)
class SearchMetadata:
    connector_identity: ConnectorIdentity
    metric_certificate: MetricCertificate
    options: SearchOptions
    policy_fingerprint: str
    policy_source: Mapping[str, Any]
    search_fingerprint: str
    core_schema_version: int = CORE_SCHEMA_VERSION


@dataclass(frozen=True)
class SearchContinuation:
    """Opaque, versioned in-memory checkpoint for a budget-stopped search."""

    checkpoint_version: str
    fingerprint: str
    payload_digest: str
    checkpoint: Any


@dataclass(frozen=True)
class SearchResult(Generic[StateT, KeyT]):
    seed: StateRecord[StateT, KeyT]
    seed_action: int
    region: tuple[StateRecord[StateT, KeyT], ...]
    boundary_counterfactuals: tuple[StateRecord[StateT, KeyT], ...]
    minimal_counterfactuals: tuple[StateRecord[StateT, KeyT], ...]
    counterfactuals: CounterfactualProjection[StateT, KeyT]
    robustness_radius: int | float | None
    best_known_radius: int | float | None
    counterfactual_existence: CounterfactualExistence
    completeness: SearchCompleteness
    stats: SearchStats
    metadata: SearchMetadata
    continuation: SearchContinuation | None
