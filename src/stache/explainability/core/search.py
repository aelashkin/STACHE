"""Budget-aware domain-neutral robustness-region and counterfactual search."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Hashable, Iterable, Mapping

from .connector import ExactActionInvariance, FormalDistanceLayer
from .identity import IdentityEncodingError, fingerprint_document
from .models import (
    ContinuationMismatchError,
    CounterfactualExistence,
    CounterfactualProjection,
    CounterfactualSelection,
    InvalidSearchOptions,
    MetricCertificationError,
    MinimumBasis,
    SearchCompleteness,
    SearchContinuation,
    SearchExtent,
    SearchInvariantError,
    SearchMetadata,
    SearchOptions,
    SearchResult,
    SearchStats,
    StateRecord,
    StopReason,
)
from .policy import (
    PolicyError,
    policy_fingerprint_for_connector,
)


CHECKPOINT_VERSION = "stache-rr-continuation-v2"


@dataclass
class _Checkpoint:
    seed: Any
    seed_key: Hashable
    seed_action: int
    states: dict[Hashable, Any]
    graph_depths: dict[Hashable, int]
    actions: dict[Hashable, int]
    ordering: dict[Hashable, tuple[Any, ...]]
    order_owners: dict[tuple[Any, ...], Hashable]
    visited: set[Hashable]
    region_keys: set[Hashable]
    boundary_keys: set[Hashable]
    graph_minimum_keys: set[Hashable] = field(default_factory=set)
    graph_minimum_depth: int | None = None
    formal_minimum_keys: set[Hashable] = field(default_factory=set)
    formal_minimum_distance: int | float | None = None
    best_known_radius: int | float | None = None
    current_layer: list[Hashable] = field(default_factory=list)
    current_depth: int = 0
    expand_index: int = 0
    next_candidates: dict[Hashable, Any] = field(default_factory=dict)
    query_order: list[Hashable] = field(default_factory=list)
    query_index: int = 0
    next_invariant: list[Hashable] = field(default_factory=list)
    phase: str = "expand"
    graph_complete: bool = False
    region_complete: bool = False
    boundary_complete: bool = False
    radius_complete: bool = False
    minima_complete: bool = False
    proven_absent: bool = False
    formal_layers: tuple[tuple[int | float, tuple[tuple[Hashable, Any], ...]], ...] = ()
    formal_layer_index: int = 0
    formal_state_index: int = 0
    formal_current_minima: set[Hashable] = field(default_factory=set)
    max_scanned_formal_distance: int | float | None = None
    states_discovered: int = 1
    states_evaluated: int = 1
    states_expanded: int = 0
    policy_queries: int = 0
    cache_hits: int = 0
    table_hits: int = 0
    model_queries: int = 0
    duplicate_discoveries: int = 0
    formal_states_scanned: int = 0
    resume_count: int = 0
    max_evaluated_graph_depth: int = 0
    max_expanded_graph_depth: int | None = None
    cache_export: Any = None


def _stable(value: Any, *, _active: set[int] | None = None) -> Any:
    """Produce injective JSON data for fingerprints and in-memory integrity."""

    if value is None or type(value) in {bool, int, float, str}:
        return value
    if _active is None:
        _active = set()
    marker = id(value)
    if marker in _active:
        raise SearchInvariantError(
            "continuation values must be acyclic for integrity encoding"
        )
    _active.add(marker)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "dataclass": (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                ),
                "fields": [
                    [
                        item.name,
                        _stable(getattr(value, item.name), _active=_active),
                    ]
                    for item in fields(value)
                ],
            }
        if isinstance(value, Mapping):
            entries = [
                [
                    _stable(key, _active=_active),
                    _stable(item, _active=_active),
                ]
                for key, item in value.items()
            ]
            entries.sort(
                key=lambda entry: json.dumps(
                    entry[0], sort_keys=True, separators=(",", ":")
                )
            )
            return {"mapping": entries}
        if isinstance(value, (set, frozenset)):
            entries = [_stable(item, _active=_active) for item in value]
            entries.sort(
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":")
                )
            )
            return {
                "frozenset" if isinstance(value, frozenset) else "set": entries
            }
        if isinstance(value, tuple):
            return {
                "tuple": [
                    _stable(item, _active=_active) for item in value
                ]
            }
        if isinstance(value, list):
            return {
                "list": [
                    _stable(item, _active=_active) for item in value
                ]
            }

        type_name = f"{type(value).__module__}.{type(value).__qualname__}"
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, Mapping):
            return {
                "object": type_name,
                "object_identity": marker,
                "attributes": _stable(attributes, _active=_active),
            }

        slot_names: list[str] = []
        for owner in type(value).__mro__:
            declared = owner.__dict__.get("__slots__", ())
            if isinstance(declared, str):
                declared = (declared,)
            slot_names.extend(
                name
                for name in declared
                if name not in {"__dict__", "__weakref__"}
                and hasattr(value, name)
            )
        if slot_names:
            return {
                "object": type_name,
                "object_identity": marker,
                "slots": [
                    [
                        name,
                        _stable(getattr(value, name), _active=_active),
                    ]
                    for name in sorted(set(slot_names))
                ],
            }
        raise SearchInvariantError(
            "continuation contains an opaque value without a stable content "
            f"encoding: {type_name}"
        )
    finally:
        _active.remove(marker)


def _checkpoint_digest(checkpoint: _Checkpoint) -> str:
    encoded = json.dumps(
        _stable(checkpoint),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _fingerprint(
    connector: Any,
    oracle: Any,
    options: SearchOptions,
    invariance: Any,
) -> str:
    return derive_search_fingerprint(
        connector,
        policy_fingerprint=str(oracle.fingerprint),
        options=options,
        invariance_fingerprint=getattr(
            invariance,
            "fingerprint",
            type(invariance).__qualname__,
        ),
        checkpoint_version=CHECKPOINT_VERSION,
    )


def derive_search_fingerprint(
    connector: Any,
    *,
    policy_fingerprint: str,
    options: SearchOptions,
    invariance_fingerprint: str = ExactActionInvariance.fingerprint,
    checkpoint_version: str = CHECKPOINT_VERSION,
) -> str:
    """Derive the canonical scientific identity of a search request."""

    payload = {
        "connector": asdict(connector.identity),
        "metric_certificate": asdict(connector.metric_certificate),
        "action_count": _connector_action_count(connector),
        "policy_fingerprint": policy_fingerprint,
        "options": dict(options.semantic_values()),
        "invariance": invariance_fingerprint,
        "checkpoint_version": checkpoint_version,
    }
    try:
        return fingerprint_document(_stable(payload))
    except IdentityEncodingError as error:
        raise SearchInvariantError(str(error)) from error


def _validate_oracle_identity(oracle: Any, connector: Any) -> None:
    source = getattr(oracle, "source_description", None)
    if not isinstance(source, Mapping):
        raise SearchInvariantError(
            "action oracle source_description must be a mapping"
        )
    try:
        derived = policy_fingerprint_for_connector(source, connector)
    except PolicyError as error:
        raise SearchInvariantError(f"invalid action oracle identity: {error}") from error
    if getattr(oracle, "fingerprint", None) != derived:
        raise SearchInvariantError(
            "action oracle fingerprint disagrees with source identity"
        )


def _canonical_state(connector: Any, state: Any) -> tuple[Any, Hashable]:
    canonical = connector.canonicalize(state)
    connector.validate_state(canonical)
    key = connector.state_key(canonical)
    try:
        hash(key)
    except TypeError as exc:
        raise SearchInvariantError("connector state_key must be hashable") from exc
    return canonical, key


def _order_key(connector: Any, key: Hashable) -> tuple[Any, ...]:
    value = connector.ordering_key(key)
    normalized = value if isinstance(value, tuple) else (value,)
    try:
        normalized < normalized
        hash(normalized)
    except (TypeError, ValueError) as exc:
        raise SearchInvariantError(
            "connector ordering keys must be hashable and totally orderable"
        ) from exc
    return normalized


def _register_state(
    checkpoint: _Checkpoint,
    connector: Any,
    state: Any,
    key: Hashable,
    *,
    graph_depth: int | None,
) -> None:
    if key in checkpoint.states:
        if checkpoint.states[key] != state:
            raise SearchInvariantError(
                f"connector state-key collision for key {key!r}"
            )
        if graph_depth is not None and key not in checkpoint.graph_depths:
            checkpoint.graph_depths[key] = graph_depth
        return

    ordering = _order_key(connector, key)
    owner = checkpoint.order_owners.get(ordering)
    if owner is not None and owner != key:
        raise SearchInvariantError(
            f"connector ordering-key collision for {ordering!r}"
        )
    checkpoint.states[key] = state
    checkpoint.ordering[key] = ordering
    checkpoint.order_owners[ordering] = key
    if graph_depth is not None:
        checkpoint.graph_depths[key] = graph_depth


def _distance(connector: Any, seed: Any, state: Any) -> int | float:
    value = connector.formal_distance(seed, state)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SearchInvariantError("formal_distance must return an int or float")
    if not math.isfinite(float(value)) or value < 0:
        raise SearchInvariantError("formal_distance must be finite and non-negative")
    return value


def _is_invariant(predicate: Any, seed_action: int, action: int) -> bool:
    if callable(predicate):
        result = predicate(seed_action, action)
    elif hasattr(predicate, "is_invariant"):
        result = predicate.is_invariant(seed_action, action)
    else:
        raise TypeError("invariance predicate must be callable")
    if type(result) is not bool:
        raise TypeError("invariance predicate must return bool")
    return result


def _connector_action_count(connector: Any) -> int:
    try:
        count = connector.action_spec.count
    except AttributeError as exc:
        raise SearchInvariantError(
            "connector must declare action_spec.count"
        ) from exc
    if type(count) is not int or count <= 0:
        raise SearchInvariantError(
            "connector action_spec.count must be a positive Python int"
        )
    return count


def _validated_action(
    connector: Any,
    action: Any,
    *,
    key: Hashable,
) -> int:
    if type(action) is not int:
        raise SearchInvariantError(
            "action oracle must return a normalized Python int "
            f"for state key {key!r}"
        )
    count = _connector_action_count(connector)
    if action < 0 or action >= count:
        raise SearchInvariantError(
            "action oracle returned an action outside the connector's "
            f"declared range [0, {count}) for state key {key!r}: {action}"
        )
    return action


def _stats_snapshot(oracle: Any) -> tuple[int, int, int, int]:
    stats = oracle.stats
    values = tuple(
        getattr(stats, field, 0)
        for field in (
            "policy_queries",
            "cache_hits",
            "table_hits",
            "model_queries",
        )
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise SearchInvariantError(
            "action oracle stats counters must be non-negative Python ints"
        )
    return values  # type: ignore[return-value]


def _stats_delta(
    before: tuple[int, int, int, int],
    after: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if any(later < earlier for earlier, later in zip(before, after)):
        raise SearchInvariantError(
            "action oracle stats counters must be monotonic"
        )
    return tuple(
        later - earlier for earlier, later in zip(before, after)
    )  # type: ignore[return-value]


def _policy_query_cost(oracle: Any, state: Any) -> int:
    cost_method = getattr(oracle, "policy_query_cost", None)
    if not callable(cost_method):
        raise SearchInvariantError(
            "action oracle must implement pure policy_query_cost(state)"
        )
    before = _stats_snapshot(oracle)
    cost = cost_method(state)
    after = _stats_snapshot(oracle)
    if after != before:
        raise SearchInvariantError(
            "action oracle policy_query_cost(state) must not mutate stats"
        )
    if type(cost) is not int or cost < 0:
        raise SearchInvariantError(
            "action oracle policy_query_cost(state) must return a "
            "non-negative Python int"
        )
    return cost


def _invoke_action(
    connector: Any,
    oracle: Any,
    state: Any,
    key: Hashable,
    *,
    expected_cost: int,
) -> tuple[int, tuple[int, int, int, int]]:
    before = _stats_snapshot(oracle)
    action = _validated_action(
        connector,
        oracle.action(state),
        key=key,
    )
    deltas = _stats_delta(before, _stats_snapshot(oracle))
    if deltas[0] != expected_cost:
        raise SearchInvariantError(
            "action oracle policy_query_cost(state) did not match the "
            "actual policy_queries delta"
        )
    return action, deltas


def _query_action(
    checkpoint: _Checkpoint,
    connector: Any,
    oracle: Any,
    state: Any,
    key: Hashable,
    options: SearchOptions,
) -> int | None:
    if key in checkpoint.actions:
        return _validated_action(
            connector,
            checkpoint.actions[key],
            key=key,
        )

    policy_cost = _policy_query_cost(oracle, state)
    if (
        options.max_policy_queries is not None
        and checkpoint.policy_queries + policy_cost
        > options.max_policy_queries
    ):
        return None

    action, deltas = _invoke_action(
        connector,
        oracle,
        state,
        key,
        expected_cost=policy_cost,
    )
    checkpoint.policy_queries += policy_cost
    checkpoint.cache_hits += deltas[1]
    checkpoint.table_hits += deltas[2]
    checkpoint.model_queries += deltas[3]
    checkpoint.actions[key] = action
    return action


def _prepare_formal_layers(
    connector: Any,
    seed: Any,
    supplied: Iterable[FormalDistanceLayer[Any]],
) -> tuple[tuple[int | float, tuple[tuple[Hashable, Any], ...]], ...]:
    declared: dict[Hashable, Any] = {}
    for raw_state in connector.declared_states():
        state, key = _canonical_state(connector, raw_state)
        existing = declared.get(key)
        if existing is not None and existing != state:
            raise SearchInvariantError(
                f"declared state-key collision for key {key!r}"
            )
        declared[key] = state

    prepared: list[tuple[int | float, tuple[tuple[Hashable, Any], ...]]] = []
    seen: set[Hashable] = set()
    previous: int | float | None = None
    for layer in supplied:
        distance = layer.distance
        if (
            isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or distance < 0
            or not math.isfinite(float(distance))
        ):
            raise MetricCertificationError(
                "formal layers must use finite non-negative distances"
            )
        if previous is not None and distance <= previous:
            raise MetricCertificationError(
                "formal layers must be strictly increasing"
            )
        previous = distance
        entries: list[tuple[Hashable, Any]] = []
        for raw_state in layer.states:
            state, key = _canonical_state(connector, raw_state)
            if key in seen:
                raise MetricCertificationError(
                    f"formal layers contain duplicate state key {key!r}"
                )
            actual = _distance(connector, seed, state)
            if actual != distance:
                raise MetricCertificationError(
                    f"formal layer distance mismatch for state {state!r}"
                )
            seen.add(key)
            entries.append((key, state))
        entries.sort(key=lambda pair: _order_key(connector, pair[0]))
        prepared.append((distance, tuple(entries)))

    if seen != set(declared):
        missing = set(declared) - seen
        extra = seen - set(declared)
        raise MetricCertificationError(
            "formal layers must cover the declared state universe exactly; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if not prepared:
        raise MetricCertificationError("formal layers may not be empty")
    return tuple(prepared)


def _graph_certifies_formal(connector: Any) -> bool:
    certificate = connector.metric_certificate
    return bool(
        certificate.every_edge_is_formal_unit
        and certificate.all_valid_formal_unit_edges_present
        and certificate.symmetric
        and certificate.connected
        and certificate.geodesic_for_formal_metric
    )


def _initialize(
    seed: Any,
    connector: Any,
    oracle: Any,
    options: SearchOptions,
    predicate: Any,
) -> tuple[_Checkpoint, str, bool]:
    canonical_seed, seed_key = _canonical_state(connector, seed)
    graph_certifies_formal = _graph_certifies_formal(connector)
    prepared_layers: tuple[
        tuple[int | float, tuple[tuple[Hashable, Any], ...]], ...
    ] = ()
    if options.minimum_basis is MinimumBasis.FORMAL_GLOBAL and not graph_certifies_formal:
        supplied = connector.formal_layers(canonical_seed)
        if supplied is None:
            raise MetricCertificationError(
                "formal_global requires a geodesic metric certificate or "
                "connector-provided formal distance layers"
            )
        if options.extent is SearchExtent.THROUGH_MINIMAL_CF:
            raise InvalidSearchOptions(
                "through_minimal_cf is unsupported for a non-geodesic "
                "formal_global search"
            )
        prepared_layers = _prepare_formal_layers(
            connector,
            canonical_seed,
            supplied,
        )

    fingerprint = _fingerprint(
        connector,
        oracle,
        options,
        predicate,
    )
    ordering = _order_key(connector, seed_key)
    seed_policy_cost = _policy_query_cost(oracle, canonical_seed)
    if (
        options.max_policy_queries is not None
        and seed_policy_cost > options.max_policy_queries
    ):
        raise InvalidSearchOptions(
            "max_policy_queries is too small to evaluate the seed"
        )
    seed_action, deltas = _invoke_action(
        connector,
        oracle,
        canonical_seed,
        seed_key,
        expected_cost=seed_policy_cost,
    )
    checkpoint = _Checkpoint(
        seed=canonical_seed,
        seed_key=seed_key,
        seed_action=seed_action,
        states={seed_key: canonical_seed},
        graph_depths={seed_key: 0},
        actions={seed_key: seed_action},
        ordering={seed_key: ordering},
        order_owners={ordering: seed_key},
        visited={seed_key},
        region_keys={seed_key},
        boundary_keys=set(),
        current_layer=[seed_key],
        formal_layers=prepared_layers,
        policy_queries=seed_policy_cost,
        cache_hits=deltas[1],
        table_hits=deltas[2],
        model_queries=deltas[3],
    )
    return checkpoint, fingerprint, graph_certifies_formal


def _validate_checkpoint(connector: Any, checkpoint: _Checkpoint) -> None:
    """Revalidate restored checkpoint structure and connector-owned identity."""

    try:
        exact_containers = {
            "states": dict,
            "graph_depths": dict,
            "actions": dict,
            "ordering": dict,
            "order_owners": dict,
            "visited": set,
            "region_keys": set,
            "boundary_keys": set,
            "graph_minimum_keys": set,
            "formal_minimum_keys": set,
            "current_layer": list,
            "next_candidates": dict,
            "query_order": list,
            "next_invariant": list,
            "formal_layers": tuple,
            "formal_current_minima": set,
        }
        for name, expected_type in exact_containers.items():
            if type(getattr(checkpoint, name)) is not expected_type:
                raise ContinuationMismatchError(
                    f"continuation checkpoint {name} must be "
                    f"{expected_type.__name__}"
                )

        canonical_seed, seed_key = _canonical_state(connector, checkpoint.seed)
        if canonical_seed != checkpoint.seed:
            raise ContinuationMismatchError(
                "continuation checkpoint seed is not canonical"
            )
        if seed_key != checkpoint.seed_key:
            raise ContinuationMismatchError(
                "continuation checkpoint seed state-key binding is invalid"
            )

        known_keys = set(checkpoint.states)
        if checkpoint.seed_key not in known_keys:
            raise ContinuationMismatchError(
                "continuation checkpoint states omit the seed key"
            )
        for stored_key, stored_state in checkpoint.states.items():
            canonical, derived_key = _canonical_state(connector, stored_state)
            if canonical != stored_state:
                raise ContinuationMismatchError(
                    "continuation checkpoint contains a non-canonical state"
                )
            if derived_key != stored_key:
                raise ContinuationMismatchError(
                    "continuation checkpoint state-key binding is invalid"
                )

        if set(checkpoint.ordering) != known_keys:
            raise ContinuationMismatchError(
                "continuation checkpoint ordering keys disagree with states"
            )
        for key, stored_order in checkpoint.ordering.items():
            if type(stored_order) is not tuple:
                raise ContinuationMismatchError(
                    "continuation checkpoint ordering values must be tuples"
                )
            if stored_order != _order_key(connector, key):
                raise ContinuationMismatchError(
                    "continuation checkpoint ordering disagrees with connector"
                )
            if checkpoint.order_owners.get(stored_order) != key:
                raise ContinuationMismatchError(
                    "continuation checkpoint order ownership is invalid"
                )
        if set(checkpoint.order_owners) != set(checkpoint.ordering.values()):
            raise ContinuationMismatchError(
                "continuation checkpoint order owners disagree with ordering"
            )

        mapping_key_sets = {
            "graph_depths": set(checkpoint.graph_depths),
            "actions": set(checkpoint.actions),
            "next_candidates": set(checkpoint.next_candidates),
        }
        for name, keys in mapping_key_sets.items():
            if not keys <= known_keys:
                raise ContinuationMismatchError(
                    f"continuation checkpoint {name} refers to unknown states"
                )
        for key, state in checkpoint.next_candidates.items():
            if checkpoint.states[key] != state:
                raise ContinuationMismatchError(
                    "continuation checkpoint candidate state binding is invalid"
                )
        for key, depth in checkpoint.graph_depths.items():
            if type(depth) is not int or depth < 0:
                raise ContinuationMismatchError(
                    "continuation checkpoint graph depths must be non-negative ints"
                )
        if checkpoint.graph_depths.get(checkpoint.seed_key) != 0:
            raise ContinuationMismatchError(
                "continuation checkpoint seed graph depth must be zero"
            )
        for key, action in checkpoint.actions.items():
            _validated_action(connector, action, key=key)
        if checkpoint.actions.get(checkpoint.seed_key) != checkpoint.seed_action:
            raise ContinuationMismatchError(
                "continuation checkpoint seed action disagrees with actions"
            )

        set_fields = (
            "visited",
            "region_keys",
            "boundary_keys",
            "graph_minimum_keys",
            "formal_minimum_keys",
            "formal_current_minima",
        )
        for name in set_fields:
            keys = getattr(checkpoint, name)
            if not keys <= known_keys:
                raise ContinuationMismatchError(
                    f"continuation checkpoint {name} refers to unknown states"
                )
        for name in ("region_keys", "boundary_keys", "graph_minimum_keys"):
            if not getattr(checkpoint, name) <= set(checkpoint.actions):
                raise ContinuationMismatchError(
                    f"continuation checkpoint {name} contains unevaluated states"
                )
        for name in ("formal_minimum_keys", "formal_current_minima"):
            if not getattr(checkpoint, name) <= set(checkpoint.actions):
                raise ContinuationMismatchError(
                    f"continuation checkpoint {name} contains unevaluated states"
                )

        for name in ("current_layer", "query_order", "next_invariant"):
            values = getattr(checkpoint, name)
            if len(values) != len(set(values)) or not set(values) <= known_keys:
                raise ContinuationMismatchError(
                    f"continuation checkpoint {name} is not a unique known-key list"
                )
        if not set(checkpoint.query_order) <= set(checkpoint.next_candidates):
            raise ContinuationMismatchError(
                "continuation checkpoint query order disagrees with candidates"
            )

        for layer in checkpoint.formal_layers:
            if type(layer) is not tuple or len(layer) != 2:
                raise ContinuationMismatchError(
                    "continuation checkpoint formal layer is malformed"
                )
            distance, entries = layer
            if (
                type(distance) not in {int, float}
                or not math.isfinite(float(distance))
                or distance < 0
                or type(entries) is not tuple
            ):
                raise ContinuationMismatchError(
                    "continuation checkpoint formal layer is malformed"
                )
            for entry in entries:
                if type(entry) is not tuple or len(entry) != 2:
                    raise ContinuationMismatchError(
                        "continuation checkpoint formal layer entry is malformed"
                    )
                entry_key, entry_state = entry
                canonical, derived_key = _canonical_state(connector, entry_state)
                if canonical != entry_state or derived_key != entry_key:
                    raise ContinuationMismatchError(
                        "continuation checkpoint formal state-key binding is invalid"
                    )

        index_limits = {
            "expand_index": len(checkpoint.current_layer),
            "query_index": len(checkpoint.query_order),
            "formal_layer_index": len(checkpoint.formal_layers),
        }
        for name, limit in index_limits.items():
            value = getattr(checkpoint, name)
            if type(value) is not int or not 0 <= value <= limit:
                raise ContinuationMismatchError(
                    f"continuation checkpoint {name} is out of range"
                )
        if checkpoint.formal_layer_index < len(checkpoint.formal_layers):
            entries = checkpoint.formal_layers[checkpoint.formal_layer_index][1]
            if not 0 <= checkpoint.formal_state_index <= len(entries):
                raise ContinuationMismatchError(
                    "continuation checkpoint formal_state_index is out of range"
                )
        elif checkpoint.formal_state_index != 0:
            raise ContinuationMismatchError(
                "continuation checkpoint formal_state_index is out of range"
            )
        if checkpoint.phase not in {"expand", "query", "formal"}:
            raise ContinuationMismatchError(
                "continuation checkpoint phase is invalid"
            )

        counter_fields = (
            "current_depth",
            "states_discovered",
            "states_evaluated",
            "states_expanded",
            "policy_queries",
            "cache_hits",
            "table_hits",
            "model_queries",
            "duplicate_discoveries",
            "formal_states_scanned",
            "resume_count",
            "max_evaluated_graph_depth",
        )
        for name in counter_fields:
            value = getattr(checkpoint, name)
            if type(value) is not int or value < 0:
                raise ContinuationMismatchError(
                    f"continuation checkpoint {name} must be a non-negative int"
                )
        for name in (
            "graph_complete",
            "region_complete",
            "boundary_complete",
            "radius_complete",
            "minima_complete",
            "proven_absent",
        ):
            if type(getattr(checkpoint, name)) is not bool:
                raise ContinuationMismatchError(
                    f"continuation checkpoint {name} must be a bool"
                )
    except ContinuationMismatchError:
        raise
    except Exception as error:
        raise ContinuationMismatchError(
            f"continuation checkpoint validation failed: {error}"
        ) from error


def _resume(
    continuation: SearchContinuation,
    seed: Any,
    connector: Any,
    oracle: Any,
    options: SearchOptions,
    predicate: Any,
) -> tuple[_Checkpoint, str, bool]:
    if continuation.checkpoint_version != CHECKPOINT_VERSION:
        raise ContinuationMismatchError(
            "checkpoint_version does not match this core version"
        )
    canonical_seed, seed_key = _canonical_state(connector, seed)
    fingerprint = _fingerprint(
        connector,
        oracle,
        options,
        predicate,
    )
    if continuation.fingerprint != fingerprint:
        raise ContinuationMismatchError(
            "fingerprint does not match the requested search"
        )
    if continuation.payload_digest != _checkpoint_digest(
        continuation.checkpoint
    ):
        raise ContinuationMismatchError(
            "continuation payload integrity check failed"
        )
    checkpoint = copy.deepcopy(continuation.checkpoint)
    if not isinstance(checkpoint, _Checkpoint):
        raise ContinuationMismatchError("continuation checkpoint payload is invalid")
    _validate_checkpoint(connector, checkpoint)
    if checkpoint.seed != canonical_seed or checkpoint.seed_key != seed_key:
        raise ContinuationMismatchError("continuation seed does not match")
    if (
        options.max_expanded is not None
        and options.max_expanded < checkpoint.states_expanded
    ):
        raise ContinuationMismatchError(
            "max_expanded may not be below the already consumed total"
        )
    if (
        options.max_policy_queries is not None
        and options.max_policy_queries < checkpoint.policy_queries
    ):
        raise ContinuationMismatchError(
            "max_policy_queries may not be below the already consumed total"
        )
    if (
        options.max_graph_depth is not None
        and options.max_graph_depth < checkpoint.max_evaluated_graph_depth
    ):
        raise ContinuationMismatchError(
            "max_graph_depth may not be below the already evaluated depth"
        )
    if checkpoint.cache_export is not None:
        oracle.restore_cache(checkpoint.cache_export)
    checkpoint.resume_count += 1
    return checkpoint, fingerprint, _graph_certifies_formal(connector)


def _expand_one(checkpoint: _Checkpoint, connector: Any) -> None:
    parent_key = checkpoint.current_layer[checkpoint.expand_index]
    parent = checkpoint.states[parent_key]
    depth = checkpoint.current_depth + 1
    local_keys: set[Hashable] = set()
    normalized: list[tuple[tuple[Any, ...], Hashable, Any]] = []
    for raw_neighbor in connector.atomic_neighbors(parent):
        state, key = _canonical_state(connector, raw_neighbor)
        if key == parent_key:
            raise SearchInvariantError(
                f"connector returned self-neighbor for key {parent_key!r}"
            )
        if key in local_keys:
            raise SearchInvariantError(
                f"connector returned duplicate neighbor key {key!r}"
            )
        local_keys.add(key)
        normalized.append((_order_key(connector, key), key, state))
    normalized.sort(key=lambda item: item[0])

    for _, key, state in normalized:
        if key in checkpoint.visited:
            existing = checkpoint.states[key]
            if existing != state:
                raise SearchInvariantError(
                    f"connector state-key collision for key {key!r}"
                )
            checkpoint.duplicate_discoveries += 1
            continue
        _register_state(
            checkpoint,
            connector,
            state,
            key,
            graph_depth=depth,
        )
        checkpoint.visited.add(key)  # visited-on-enqueue
        checkpoint.next_candidates[key] = state
        checkpoint.states_discovered += 1

    checkpoint.expand_index += 1
    checkpoint.states_expanded += 1
    checkpoint.max_expanded_graph_depth = checkpoint.current_depth


def _finish_graph(
    checkpoint: _Checkpoint,
    connector: Any,
    options: SearchOptions,
    graph_certifies_formal: bool,
) -> StopReason | None:
    checkpoint.graph_complete = True
    checkpoint.region_complete = True
    checkpoint.boundary_complete = True

    if options.minimum_basis is MinimumBasis.GRAPH_BOUNDARY:
        if checkpoint.graph_minimum_depth is not None:
            checkpoint.radius_complete = True
            checkpoint.minima_complete = True
        elif (
            connector.metric_certificate.connected
            and connector.metric_certificate.symmetric
        ):
            checkpoint.proven_absent = True
            checkpoint.radius_complete = True
            checkpoint.minima_complete = True
        return StopReason.COMPLETE

    if graph_certifies_formal:
        if checkpoint.graph_minimum_depth is not None:
            _project_certified_graph_minimum(checkpoint, connector)
            checkpoint.radius_complete = True
            checkpoint.minima_complete = True
        else:
            checkpoint.proven_absent = True
            checkpoint.radius_complete = True
            checkpoint.minima_complete = True
        return StopReason.COMPLETE

    checkpoint.phase = "formal"
    return None


def _project_certified_graph_minimum(
    checkpoint: _Checkpoint,
    connector: Any,
) -> None:
    """Project an observed geodesic graph minimum into formal result fields."""

    if checkpoint.graph_minimum_depth is None:
        return
    if not checkpoint.graph_minimum_keys:
        raise SearchInvariantError(
            "graph minimum depth exists without any minimum state"
        )
    distances = {
        _distance(connector, checkpoint.seed, checkpoint.states[key])
        for key in checkpoint.graph_minimum_keys
    }
    if len(distances) != 1:
        raise MetricCertificationError(
            "geodesic certificate contradicted by boundary distances"
        )
    formal_distance = distances.pop()
    expected = (
        checkpoint.graph_minimum_depth
        * connector.metric_certificate.formal_unit
    )
    if not math.isclose(float(formal_distance), float(expected)):
        raise MetricCertificationError(
            "geodesic certificate contradicted by graph depth and formal unit"
        )
    checkpoint.formal_minimum_keys = set(checkpoint.graph_minimum_keys)
    checkpoint.formal_minimum_distance = formal_distance


def _run_graph(
    checkpoint: _Checkpoint,
    connector: Any,
    oracle: Any,
    options: SearchOptions,
    predicate: Any,
    graph_certifies_formal: bool,
) -> StopReason | None:
    while not checkpoint.graph_complete:
        if checkpoint.phase == "expand":
            if checkpoint.expand_index < len(checkpoint.current_layer):
                if (
                    options.max_graph_depth is not None
                    and checkpoint.current_depth >= options.max_graph_depth
                ):
                    return StopReason.MAX_GRAPH_DEPTH
                if (
                    options.max_expanded is not None
                    and checkpoint.states_expanded >= options.max_expanded
                ):
                    return StopReason.MAX_EXPANDED
                _expand_one(checkpoint, connector)
                continue

            if not checkpoint.next_candidates:
                return _finish_graph(
                    checkpoint,
                    connector,
                    options,
                    graph_certifies_formal,
                )
            checkpoint.query_order = sorted(
                checkpoint.next_candidates,
                key=lambda key: checkpoint.ordering[key],
            )
            checkpoint.query_index = 0
            checkpoint.next_invariant = []
            checkpoint.phase = "query"
            continue

        if checkpoint.phase != "query":
            return None

        next_depth = checkpoint.current_depth + 1
        if checkpoint.query_index < len(checkpoint.query_order):
            key = checkpoint.query_order[checkpoint.query_index]
            state = checkpoint.states[key]
            action = _query_action(
                checkpoint,
                connector,
                oracle,
                state,
                key,
                options,
            )
            if action is None:
                return StopReason.MAX_POLICY_QUERIES
            checkpoint.states_evaluated += 1
            checkpoint.max_evaluated_graph_depth = max(
                checkpoint.max_evaluated_graph_depth,
                next_depth,
            )
            if _is_invariant(predicate, checkpoint.seed_action, action):
                checkpoint.region_keys.add(key)
                checkpoint.next_invariant.append(key)
            else:
                checkpoint.boundary_keys.add(key)
                if checkpoint.graph_minimum_depth is None:
                    checkpoint.graph_minimum_depth = next_depth
                    checkpoint.graph_minimum_keys = {key}
                    if options.minimum_basis is MinimumBasis.GRAPH_BOUNDARY:
                        checkpoint.radius_complete = True
                    elif graph_certifies_formal:
                        checkpoint.radius_complete = True
                elif checkpoint.graph_minimum_depth == next_depth:
                    checkpoint.graph_minimum_keys.add(key)

                if options.minimum_basis is MinimumBasis.GRAPH_BOUNDARY:
                    bound: int | float = float(next_depth)
                else:
                    bound = _distance(connector, checkpoint.seed, state)
                if (
                    checkpoint.best_known_radius is None
                    or bound < checkpoint.best_known_radius
                ):
                    checkpoint.best_known_radius = bound
            checkpoint.query_index += 1
            continue

        layer_had_counterfactual = (
            checkpoint.graph_minimum_depth == next_depth
        )
        if layer_had_counterfactual:
            if (
                options.minimum_basis is MinimumBasis.GRAPH_BOUNDARY
                or graph_certifies_formal
            ):
                checkpoint.minima_complete = True
            if options.extent is SearchExtent.THROUGH_MINIMAL_CF:
                return StopReason.THROUGH_MINIMAL

        if not checkpoint.next_invariant:
            return _finish_graph(
                checkpoint,
                connector,
                options,
                graph_certifies_formal,
            )

        checkpoint.current_layer = sorted(
            checkpoint.next_invariant,
            key=lambda key: checkpoint.ordering[key],
        )
        checkpoint.current_depth = next_depth
        checkpoint.expand_index = 0
        checkpoint.next_candidates = {}
        checkpoint.query_order = []
        checkpoint.query_index = 0
        checkpoint.next_invariant = []
        checkpoint.phase = "expand"
    return None


def _run_formal(
    checkpoint: _Checkpoint,
    connector: Any,
    oracle: Any,
    options: SearchOptions,
    predicate: Any,
) -> StopReason | None:
    while checkpoint.formal_layer_index < len(checkpoint.formal_layers):
        distance, entries = checkpoint.formal_layers[checkpoint.formal_layer_index]
        if checkpoint.formal_state_index < len(entries):
            key, state = entries[checkpoint.formal_state_index]
            newly_discovered = key not in checkpoint.states
            _register_state(
                checkpoint,
                connector,
                state,
                key,
                graph_depth=checkpoint.graph_depths.get(key),
            )
            if newly_discovered:
                checkpoint.states_discovered += 1
            newly_evaluated = key not in checkpoint.actions
            action = _query_action(
                checkpoint,
                connector,
                oracle,
                state,
                key,
                options,
            )
            if action is None:
                return StopReason.MAX_POLICY_QUERIES
            if newly_evaluated:
                checkpoint.states_evaluated += 1
            checkpoint.formal_states_scanned += 1
            checkpoint.formal_state_index += 1
            checkpoint.max_scanned_formal_distance = distance
            if not _is_invariant(predicate, checkpoint.seed_action, action):
                checkpoint.formal_current_minima.add(key)
                checkpoint.radius_complete = True
                checkpoint.formal_minimum_distance = distance
                if (
                    checkpoint.best_known_radius is None
                    or distance < checkpoint.best_known_radius
                ):
                    checkpoint.best_known_radius = distance
            continue

        if checkpoint.formal_current_minima:
            checkpoint.formal_minimum_keys = set(
                checkpoint.formal_current_minima
            )
            checkpoint.minima_complete = True
            return StopReason.COMPLETE
        checkpoint.formal_layer_index += 1
        checkpoint.formal_state_index = 0
        checkpoint.formal_current_minima = set()

    checkpoint.proven_absent = True
    checkpoint.radius_complete = True
    checkpoint.minima_complete = True
    return StopReason.COMPLETE


def _remaining_frontier(checkpoint: _Checkpoint, reason: StopReason) -> int:
    if reason in {StopReason.COMPLETE, StopReason.THROUGH_MINIMAL}:
        return 0
    if checkpoint.phase == "expand":
        return (
            len(checkpoint.current_layer) - checkpoint.expand_index
            + len(checkpoint.next_candidates)
        )
    if checkpoint.phase == "query":
        return (
            len(checkpoint.query_order) - checkpoint.query_index
            + len(checkpoint.next_invariant)
        )
    if checkpoint.phase == "formal" and checkpoint.formal_layers:
        _, entries = checkpoint.formal_layers[checkpoint.formal_layer_index]
        return len(entries) - checkpoint.formal_state_index
    return 0


def _record(
    checkpoint: _Checkpoint,
    connector: Any,
    key: Hashable,
    *,
    discovery_source: str = "graph",
) -> StateRecord[Any, Hashable]:
    state = checkpoint.states[key]
    return StateRecord(
        state=state,
        key=key,
        action=checkpoint.actions[key],
        graph_depth=checkpoint.graph_depths.get(key),
        formal_distance=_distance(connector, checkpoint.seed, state),
        discovery_source=discovery_source,
    )


def _ordered_records(
    checkpoint: _Checkpoint,
    connector: Any,
    keys: Iterable[Hashable],
    *,
    formal: bool = False,
) -> tuple[StateRecord[Any, Hashable], ...]:
    def sort_key(key: Hashable) -> tuple[Any, ...]:
        if formal:
            return (
                float(_distance(connector, checkpoint.seed, checkpoint.states[key])),
                checkpoint.ordering[key],
            )
        depth = checkpoint.graph_depths.get(key)
        return (
            depth is None,
            depth if depth is not None else math.inf,
            checkpoint.ordering[key],
        )

    return tuple(
        _record(
            checkpoint,
            connector,
            key,
            discovery_source=(
                "formal"
                if formal and key not in checkpoint.graph_depths
                else "graph"
            ),
        )
        for key in sorted(keys, key=sort_key)
    )


def _build_result(
    checkpoint: _Checkpoint,
    connector: Any,
    oracle: Any,
    options: SearchOptions,
    fingerprint: str,
    reason: StopReason,
) -> SearchResult[Any, Hashable]:
    if options.minimum_basis is MinimumBasis.FORMAL_GLOBAL:
        minimum_keys = checkpoint.formal_minimum_keys
        if (
            checkpoint.radius_complete
            and not checkpoint.minima_complete
            and checkpoint.formal_current_minima
        ):
            minimum_keys = checkpoint.formal_current_minima
        formal_order = True
        radius = (
            checkpoint.formal_minimum_distance
            if checkpoint.radius_complete
            else None
        )
    else:
        minimum_keys = checkpoint.graph_minimum_keys
        formal_order = False
        radius = (
            float(checkpoint.graph_minimum_depth)
            if checkpoint.radius_complete
            and checkpoint.graph_minimum_depth is not None
            else None
        )

    if checkpoint.proven_absent:
        existence = CounterfactualExistence.PROVEN_ABSENT
    elif checkpoint.boundary_keys or checkpoint.formal_current_minima or checkpoint.formal_minimum_keys:
        existence = CounterfactualExistence.FOUND
    else:
        existence = CounterfactualExistence.UNKNOWN

    region = _ordered_records(
        checkpoint,
        connector,
        checkpoint.region_keys,
    )
    boundary = _ordered_records(
        checkpoint,
        connector,
        checkpoint.boundary_keys,
    )
    minima = _ordered_records(
        checkpoint,
        connector,
        minimum_keys,
        formal=formal_order,
    )
    projection = CounterfactualProjection(
        minimal=(
            minima
            if options.counterfactuals
            in {CounterfactualSelection.MINIMAL, CounterfactualSelection.BOTH}
            else None
        ),
        boundary=(
            boundary
            if options.counterfactuals
            in {CounterfactualSelection.BOUNDARY, CounterfactualSelection.BOTH}
            else None
        ),
    )
    continuation: SearchContinuation | None = None
    if reason in {
        StopReason.MAX_EXPANDED,
        StopReason.MAX_POLICY_QUERIES,
        StopReason.MAX_GRAPH_DEPTH,
    }:
        checkpoint.cache_export = oracle.export_cache()
        snapshot = copy.deepcopy(checkpoint)
        continuation = SearchContinuation(
            checkpoint_version=CHECKPOINT_VERSION,
            fingerprint=fingerprint,
            payload_digest=_checkpoint_digest(snapshot),
            checkpoint=snapshot,
        )

    source = oracle.source_description
    if not isinstance(source, Mapping):
        source = {"source": str(source)}
    metadata = SearchMetadata(
        connector_identity=connector.identity,
        metric_certificate=connector.metric_certificate,
        options=options,
        policy_fingerprint=str(oracle.fingerprint),
        policy_source=dict(source),
        search_fingerprint=fingerprint,
    )
    stats = SearchStats(
        states_discovered=checkpoint.states_discovered,
        states_evaluated=checkpoint.states_evaluated,
        states_expanded=checkpoint.states_expanded,
        policy_queries=checkpoint.policy_queries,
        cache_hits=checkpoint.cache_hits,
        table_hits=checkpoint.table_hits,
        model_queries=checkpoint.model_queries,
        duplicate_discoveries=checkpoint.duplicate_discoveries,
        formal_states_scanned=checkpoint.formal_states_scanned,
        resume_count=checkpoint.resume_count,
    )
    completeness = SearchCompleteness(
        region_complete=checkpoint.region_complete,
        boundary_complete=checkpoint.boundary_complete,
        radius_complete=checkpoint.radius_complete,
        minimal_counterfactuals_complete=checkpoint.minima_complete,
        max_evaluated_graph_depth=checkpoint.max_evaluated_graph_depth,
        max_expanded_graph_depth=checkpoint.max_expanded_graph_depth,
        max_scanned_formal_distance=checkpoint.max_scanned_formal_distance,
        remaining_frontier_size=_remaining_frontier(checkpoint, reason),
        stop_reason=reason,
    )
    return SearchResult(
        seed=_record(checkpoint, connector, checkpoint.seed_key),
        seed_action=checkpoint.seed_action,
        region=region,
        boundary_counterfactuals=boundary,
        minimal_counterfactuals=minima,
        counterfactuals=projection,
        robustness_radius=radius,
        best_known_radius=checkpoint.best_known_radius,
        counterfactual_existence=existence,
        completeness=completeness,
        stats=stats,
        metadata=metadata,
        continuation=continuation,
    )


def compute_rr(
    seed: Any,
    connector: Any,
    oracle: Any,
    options: SearchOptions | None = None,
    *,
    invariance: Any | None = None,
    continuation: SearchContinuation | None = None,
) -> SearchResult[Any, Hashable]:
    """Compute an RR and counterfactual assessment over a connector graph.

    Resource ceilings are total ceilings.  A returned continuation may be
    supplied with larger ceilings (or no ceilings) without changing scientific
    options.  Policy actions are resolved once per canonical key.
    """

    if options is None:
        options = SearchOptions()
    if not isinstance(options, SearchOptions):
        raise TypeError("options must be a SearchOptions instance")
    predicate = ExactActionInvariance() if invariance is None else invariance
    if type(predicate) is not ExactActionInvariance:
        raise InvalidSearchOptions(
            "Phase 1 supports exact action invariance only"
        )
    _connector_action_count(connector)
    _validate_oracle_identity(oracle, connector)

    if continuation is None:
        checkpoint, fingerprint, graph_certifies_formal = _initialize(
            seed,
            connector,
            oracle,
            options,
            predicate,
        )
    else:
        checkpoint, fingerprint, graph_certifies_formal = _resume(
            continuation,
            seed,
            connector,
            oracle,
            options,
            predicate,
        )

    reason = _run_graph(
        checkpoint,
        connector,
        oracle,
        options,
        predicate,
        graph_certifies_formal,
    )
    if reason is None and checkpoint.phase == "formal":
        reason = _run_formal(
            checkpoint,
            connector,
            oracle,
            options,
            predicate,
        )
    if reason is None:
        raise SearchInvariantError("search stopped without a stop reason")
    if (
        graph_certifies_formal
        and options.minimum_basis is MinimumBasis.FORMAL_GLOBAL
    ):
        _project_certified_graph_minimum(checkpoint, connector)
    return _build_result(
        checkpoint,
        connector,
        oracle,
        options,
        fingerprint,
        reason,
    )
