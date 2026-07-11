"""Taxi-facing entry points for the domain-neutral RR search core.

``compute_taxi_rr`` is the typed API.  The original helpers remain as narrow,
deprecated compatibility shims so existing notebooks and scripts can migrate
without preserving a second Taxi-specific search implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any
import warnings

import numpy as np

from stache.explainability.connectors.taxi import TaxiConnector, TaxiState
from stache.explainability.core.models import (
    SearchContinuation,
    SearchOptions,
    SearchResult,
)
from stache.explainability.core.policy import (
    ModelActionOracle,
    TableActionOracle,
    TableThenModelActionOracle,
)
from stache.explainability.core.search import compute_rr


def compute_taxi_rr(
    seed: TaxiState,
    *,
    model: object | None = None,
    policy_table: Mapping[int, object] | None = None,
    model_fingerprint: str | None = None,
    table_fingerprint: str | None = None,
    options: SearchOptions | None = None,
    continuation: SearchContinuation | None = None,
) -> SearchResult[TaxiState, TaxiState]:
    """Compute a Taxi RR through the single generic search implementation.

    Exactly one explicit source strategy is selected from the supplied values:
    a strict table, a model, or table-then-model fallback.  A model fingerprint
    is required by the modern API because object identity is not evidence of
    model contents; callers loading a checkpoint should hash that checkpoint.
    Table fingerprints are derived deterministically when omitted.
    """

    connector = TaxiConnector()
    if model is None and policy_table is None:
        raise ValueError("compute_taxi_rr requires a model or policy_table")

    if model is None:
        assert policy_table is not None
        oracle = TableActionOracle(
            connector,
            policy_table,
            source_fingerprint=table_fingerprint,
        )
    else:
        if not isinstance(model_fingerprint, str) or not model_fingerprint.strip():
            raise ValueError(
                "model_fingerprint is required when a model policy source is used"
            )
        if policy_table is None:
            oracle = ModelActionOracle(
                connector,
                model,
                source_fingerprint=model_fingerprint,
            )
        else:
            oracle = TableThenModelActionOracle(
                connector,
                policy_table,
                model,
                table_fingerprint=table_fingerprint,
                model_fingerprint=model_fingerprint,
            )

    result = compute_rr(
        seed,
        connector,
        oracle,
        options,
        continuation=continuation,
    )
    return result


def translate_tuple_to_onehot(env: object, state: TaxiState) -> np.ndarray:
    """Deprecated one-row adapter around ``TaxiConnector.encode_observation``.

    ``env`` is retained only for source compatibility.  When it exposes a
    discrete observation count, that count must still be 500.
    """

    warnings.warn(
        "translate_tuple_to_onehot is deprecated; use "
        "TaxiConnector.encode_observation instead",
        DeprecationWarning,
        stacklevel=2,
    )
    _validate_legacy_env(env)
    observation = TaxiConnector().encode_observation(state)
    return observation[np.newaxis, :]


def get_neighbors_taxi(state: TaxiState) -> list[TaxiState]:
    """Deprecated list adapter around ``TaxiConnector.atomic_neighbors``."""

    warnings.warn(
        "get_neighbors_taxi is deprecated; use TaxiConnector.atomic_neighbors "
        "instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return list(TaxiConnector().atomic_neighbors(state))


def compute_rr_taxi(
    seed_f: TaxiState,
    model: object,
    env: object,
    precomputed_sa: Mapping[int, object] | None = None,
) -> dict[str, Any]:
    """Deprecated compatibility wrapper returning the historical mapping.

    The old implementation evaluated the seed with the model even when a
    precomputed table was supplied.  The compatibility wrapper deliberately
    fixes that inconsistency: ``precomputed_sa`` is an explicit table-then-model
    source for every state, including the seed.  Scientific computation is
    delegated to :func:`compute_taxi_rr`; only the return-shape conversion and
    elapsed wall-clock measurement remain here.
    """

    warnings.warn(
        "compute_rr_taxi is deprecated; use compute_taxi_rr and SearchResult "
        "instead",
        DeprecationWarning,
        stacklevel=2,
    )
    _validate_legacy_env(env)
    started = time.perf_counter()
    result = compute_taxi_rr(
        seed_f,
        model=model,
        policy_table=precomputed_sa,
        model_fingerprint=_legacy_model_fingerprint(model),
    )
    elapsed = time.perf_counter() - started

    rr_depths: dict[TaxiState, int] = {}
    for record in result.region:
        if record.graph_depth is None:
            raise RuntimeError(
                "Taxi graph-region records must carry a graph depth"
            )
        rr_depths[record.state] = record.graph_depth

    counterfactuals: list[tuple[TaxiState, int, int]] = []
    for record in result.boundary_counterfactuals:
        if record.graph_depth is None:
            raise RuntimeError(
                "Taxi graph-boundary records must carry a graph depth"
            )
        counterfactuals.append(
            (record.state, record.action, record.graph_depth)
        )

    rr_tuple_set = set(rr_depths)
    return {
        "rr_tuple_set": rr_tuple_set,
        "rr_depths": rr_depths,
        "counterfactuals": counterfactuals,
        "initial_action": result.seed_action,
        "stats": {
            "region_size": len(rr_tuple_set),
            "visited": result.stats.states_discovered,
            "opened": result.stats.states_evaluated,
            "elapsed": elapsed,
        },
    }


def _validate_legacy_env(env: object) -> None:
    """Reject a clearly incompatible legacy Taxi environment when inspectable."""

    base_env = getattr(env, "unwrapped", env)
    observation_space = getattr(base_env, "observation_space", None)
    state_count = getattr(observation_space, "n", None)
    if state_count is not None and state_count != 500:
        raise ValueError(
            "legacy Taxi environment must expose Discrete(500), "
            f"got n={state_count!r}"
        )


def _legacy_model_fingerprint(model: object) -> str:
    """Mark the shim's unverifiable in-memory model identity explicitly."""

    model_type = type(model)
    return (
        "legacy-unverified-model:"
        f"{model_type.__module__}.{model_type.__qualname__}"
    )


__all__ = [
    "compute_taxi_rr",
    "compute_rr_taxi",
    "get_neighbors_taxi",
    "translate_tuple_to_onehot",
]
