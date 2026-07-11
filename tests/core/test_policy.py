"""Behavioral contracts for discrete policy action oracles.

The fakes in this module deliberately implement only the connector and model
surface used by the policy layer.  In particular, they do not import Gymnasium
or Stable-Baselines3, so failures identify our boundary contract rather than a
third-party runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest

from stache.explainability.core.policy import (
    ActionShapeError,
    ActionValidationError,
    CacheRestoreError,
    ModelActionOracle,
    ModelCompatibilityError,
    OracleStats,
    TableActionOracle,
    TableThenModelActionOracle,
    UnknownTableKeyError,
    normalize_discrete_action,
)


@dataclass(frozen=True)
class _ObservationSpec:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _DiscreteActionSpec:
    count: int


class TinyPolicyConnector:
    """Small connector with distinct canonical and policy-table keys."""

    _observations = {
        "seed": np.array([0, 0], dtype=np.int64),
        "left": np.array([0, 1], dtype=np.int64),
        "right": np.array([1, 0], dtype=np.int64),
    }

    def __init__(
        self,
        *,
        observation_shape: tuple[int, ...] = (2,),
        observation_dtype: str = "int64",
        action_count: int = 3,
        encoded_overrides: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.observation_spec = _ObservationSpec(
            shape=observation_shape,
            dtype=observation_dtype,
        )
        self.action_spec = _DiscreteActionSpec(count=action_count)
        self._encoded_overrides = encoded_overrides or {}

    def canonicalize(self, state: str) -> str:
        if not isinstance(state, str):
            raise TypeError("tiny states must be strings")
        return state.strip().lower()

    def validate_state(self, state: str) -> None:
        if state not in self._observations:
            raise ValueError(f"unknown tiny state: {state!r}")

    def state_key(self, state: str) -> str:
        canonical = self.canonicalize(state)
        self.validate_state(canonical)
        return f"state:{canonical}"

    def encode_observation(self, state: str) -> np.ndarray:
        canonical = self.canonicalize(state)
        self.validate_state(canonical)
        observation = self._encoded_overrides.get(
            canonical,
            self._observations[canonical],
        )
        return np.array(observation, copy=True)

    def policy_lookup_key(self, state: str) -> str:
        canonical = self.canonicalize(state)
        self.validate_state(canonical)
        return f"policy:{canonical}"


@dataclass(frozen=True)
class DuckObservationSpace:
    """Gymnasium-free observation-space declaration used for compatibility checks."""

    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class DuckDiscreteSpace:
    """Gymnasium-free discrete action-space declaration."""

    n: int


class DeterministicModel:
    """Small SB3-shaped model fake whose calls are observable to the tests."""

    def __init__(
        self,
        actions: dict[tuple[int, ...], int],
        *,
        observation_shape: tuple[int, ...] = (2,),
        observation_dtype: np.dtype | type[np.generic] = np.dtype("int64"),
        action_count: int = 3,
        output: Callable[[int], object] | None = None,
    ) -> None:
        self.observation_space = DuckObservationSpace(
            shape=observation_shape,
            dtype=np.dtype(observation_dtype),
        )
        self.action_space = DuckDiscreteSpace(n=action_count)
        self._actions = actions
        self._output = output or (
            lambda action: np.array([action], dtype=np.int64)
        )
        self.calls: list[tuple[np.ndarray, bool]] = []

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> tuple[object, None]:
        copied = np.array(observation, copy=True)
        self.calls.append((copied, deterministic))
        key = tuple(int(value) for value in copied.tolist())
        return self._output(self._actions[key]), None


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1, id="python-int"),
        pytest.param(np.int64(1), id="numpy-integer-scalar"),
        pytest.param(
            np.array(1, dtype=np.int32),
            id="zero-dimensional-integer-array",
        ),
        pytest.param(
            np.array([1], dtype=np.int64),
            id="one-element-integer-vector",
        ),
    ],
)
def test_normalize_discrete_action_accepts_exact_integer_scalar_forms(
    value: object,
) -> None:
    action = normalize_discrete_action(value, action_count=3)

    assert action == 1
    assert type(action) is int


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="python-bool"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param(1.0, id="python-float"),
        pytest.param(np.float64(1.0), id="numpy-float-scalar"),
        pytest.param(np.array(1.0), id="zero-dimensional-float-array"),
        pytest.param("1", id="string"),
        pytest.param(np.array(["1"]), id="one-element-string-vector"),
    ],
)
def test_normalize_discrete_action_rejects_non_integer_values(value: object) -> None:
    with pytest.raises(ActionValidationError, match="integer"):
        normalize_discrete_action(value, action_count=3)


@pytest.mark.parametrize(
    "value, expected_shape",
    [
        pytest.param(np.array([], dtype=np.int64), "\\(0,\\)", id="empty"),
        pytest.param(np.array([0, 1]), "\\(2,\\)", id="two-actions"),
        pytest.param(np.array([[1]]), "\\(1, 1\\)", id="nested-size-one"),
    ],
)
def test_normalize_discrete_action_rejects_non_scalar_array_shapes(
    value: np.ndarray,
    expected_shape: str,
) -> None:
    with pytest.raises(ActionShapeError, match=expected_shape):
        normalize_discrete_action(value, action_count=3)


@pytest.mark.parametrize("value", [-1, 3, np.int64(4), np.array([-1])])
def test_normalize_discrete_action_rejects_actions_outside_declared_space(
    value: object,
) -> None:
    with pytest.raises(ActionValidationError, match="range"):
        normalize_discrete_action(value, action_count=3)


@pytest.mark.parametrize("action_count", [0, -1, True, 3.0])
def test_normalize_discrete_action_rejects_invalid_action_count(
    action_count: object,
) -> None:
    with pytest.raises(ActionValidationError, match="action_count"):
        normalize_discrete_action(0, action_count=action_count)  # type: ignore[arg-type]


def test_table_oracle_uses_policy_key_and_one_cache_for_canonical_states() -> None:
    connector = TinyPolicyConnector()
    oracle = TableActionOracle(
        connector,
        {"policy:seed": np.int64(2)},
        source_fingerprint="table-v1",
    )

    assert oracle.action(" SEED ") == 2
    assert oracle.has_cached("seed")
    assert oracle.action("seed") == 2

    assert oracle.cache_size == 1
    assert oracle.stats == OracleStats(
        policy_queries=1,
        cache_hits=1,
        table_hits=1,
        model_queries=0,
    )
    assert oracle.fingerprint == "table-v1"
    assert oracle.source_description["source"] == "table"


def test_table_oracle_reports_unknown_keys_without_model_fallback() -> None:
    oracle = TableActionOracle(
        TinyPolicyConnector(),
        {"policy:seed": 0},
        source_fingerprint="table-v1",
    )

    with pytest.raises(UnknownTableKeyError, match="policy:right"):
        oracle.action("right")


@pytest.mark.parametrize(
    "invalid_action",
    [
        pytest.param(True, id="boolean"),
        pytest.param(1.0, id="float"),
        pytest.param(np.array([0, 1]), id="batched"),
        pytest.param(3, id="out-of-range"),
    ],
)
def test_table_oracle_rejects_malformed_values_during_construction(
    invalid_action: object,
) -> None:
    with pytest.raises((ActionShapeError, ActionValidationError), match="policy:seed"):
        TableActionOracle(
            TinyPolicyConnector(),
            {"policy:seed": invalid_action},
        )


def test_implicit_table_fingerprint_is_deterministic_across_mapping_order() -> None:
    connector = TinyPolicyConnector()
    first = TableActionOracle(
        connector,
        {"policy:seed": 0, "policy:left": np.int64(1)},
    )
    second = TableActionOracle(
        connector,
        {"policy:left": 1, "policy:seed": np.int32(0)},
    )

    assert first.fingerprint == second.fingerprint
    assert isinstance(first.fingerprint, str)
    assert first.fingerprint


def test_cache_export_restore_avoids_requerying_the_policy_source() -> None:
    connector = TinyPolicyConnector()
    source = TableActionOracle(
        connector,
        {"policy:seed": 0, "policy:left": 1},
        source_fingerprint="table-v1",
    )
    assert source.action("seed") == 0
    assert source.action("left") == 1

    restored = TableActionOracle(
        connector,
        {"policy:seed": 2, "policy:left": 2},
        source_fingerprint="table-v1",
    )
    restored.restore_cache(source.export_cache())

    assert restored.cache_size == 2
    assert restored.action("seed") == 0
    assert restored.action("left") == 1
    assert restored.stats == OracleStats(
        policy_queries=0,
        cache_hits=2,
        table_hits=0,
        model_queries=0,
    )


def test_cache_restore_rejects_malformed_records_with_a_specific_error() -> None:
    oracle = TableActionOracle(
        TinyPolicyConnector(),
        {"policy:seed": 0},
        source_fingerprint="table-v1",
    )

    with pytest.raises(CacheRestoreError, match="cache"):
        oracle.restore_cache(({"unexpected": "record"},))


def test_model_oracle_predicts_deterministically_and_caches_normalized_action() -> None:
    connector = TinyPolicyConnector()
    model = DeterministicModel({(0, 1): 1})
    oracle = ModelActionOracle(
        connector,
        model,
        source_fingerprint="model-v1",
    )

    assert oracle.action(" LEFT ") == 1
    assert oracle.action("left") == 1

    assert len(model.calls) == 1
    observation, deterministic = model.calls[0]
    np.testing.assert_array_equal(observation, np.array([0, 1]))
    assert deterministic is True
    assert oracle.stats == OracleStats(
        policy_queries=1,
        cache_hits=1,
        table_hits=0,
        model_queries=1,
    )
    assert oracle.fingerprint == "model-v1"
    assert oracle.source_description["source"] == "model"


@pytest.mark.parametrize(
    "model_kwargs, diagnostic",
    [
        pytest.param(
            {"observation_shape": (3,)},
            "observation.*shape",
            id="observation-shape",
        ),
        pytest.param(
            {"observation_dtype": np.dtype("float32")},
            "observation.*dtype",
            id="observation-dtype",
        ),
        pytest.param(
            {"action_count": 4},
            "action.*space",
            id="action-count",
        ),
    ],
)
def test_model_oracle_rejects_declared_space_mismatches(
    model_kwargs: dict[str, object],
    diagnostic: str,
) -> None:
    model = DeterministicModel({(0, 0): 0}, **model_kwargs)  # type: ignore[arg-type]

    with pytest.raises(ModelCompatibilityError, match=diagnostic):
        ModelActionOracle(
            TinyPolicyConnector(),
            model,
            source_fingerprint="model-v1",
        )


@pytest.mark.parametrize(
    "encoded, diagnostic",
    [
        pytest.param(
            np.array([0, 0, 0], dtype=np.int64),
            "encoded observation.*shape",
            id="encoded-shape",
        ),
        pytest.param(
            np.array([0, 0], dtype=np.float32),
            "encoded observation.*dtype",
            id="encoded-dtype",
        ),
    ],
)
def test_model_oracle_rejects_connector_encoding_that_violates_its_spec(
    encoded: np.ndarray,
    diagnostic: str,
) -> None:
    connector = TinyPolicyConnector(encoded_overrides={"seed": encoded})
    oracle = ModelActionOracle(
        connector,
        DeterministicModel({(0, 0): 0}),
        source_fingerprint="model-v1",
    )

    with pytest.raises(ModelCompatibilityError, match=diagnostic):
        oracle.action("seed")


@pytest.mark.parametrize(
    "output, expected_error",
    [
        pytest.param(
            lambda action: np.array([[action]], dtype=np.int64),
            ActionShapeError,
            id="nested-action-array",
        ),
        pytest.param(
            lambda _action: np.array([3], dtype=np.int64),
            ActionValidationError,
            id="out-of-range-action",
        ),
    ],
)
def test_model_oracle_rejects_invalid_model_action_output(
    output: Callable[[int], object],
    expected_error: type[Exception],
) -> None:
    oracle = ModelActionOracle(
        TinyPolicyConnector(),
        DeterministicModel({(0, 0): 0}, output=output),
        source_fingerprint="model-v1",
    )

    with pytest.raises(expected_error):
        oracle.action("seed")


def test_table_then_model_uses_table_for_seed_and_model_only_on_missing_key() -> None:
    connector = TinyPolicyConnector()
    model = DeterministicModel({(0, 0): 2, (1, 0): 1})
    oracle = TableThenModelActionOracle(
        connector,
        {"policy:seed": 0},
        model,
        table_fingerprint="table-v1",
        model_fingerprint="model-v1",
    )

    # A search queries its seed through the same action() surface as candidates.
    assert oracle.action("seed") == 0
    assert model.calls == []

    assert oracle.action("right") == 1
    assert oracle.action("right") == 1
    assert len(model.calls) == 1
    assert model.calls[0][1] is True
    assert oracle.stats == OracleStats(
        policy_queries=2,
        cache_hits=1,
        table_hits=1,
        model_queries=1,
    )
    assert oracle.source_description["source"] == "table_then_model"
    assert oracle.source_description["table_fingerprint"] == "table-v1"
    assert oracle.source_description["model_fingerprint"] == "model-v1"
    assert isinstance(oracle.fingerprint, str)
    assert oracle.fingerprint


def test_table_then_model_rejects_malformed_table_instead_of_falling_back() -> None:
    model = DeterministicModel({(0, 0): 1})

    with pytest.raises(ActionShapeError, match="policy:seed"):
        TableThenModelActionOracle(
            TinyPolicyConnector(),
            {"policy:seed": np.array([[0]], dtype=np.int64)},
            model,
            model_fingerprint="model-v1",
        )

    assert model.calls == []
