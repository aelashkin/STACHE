"""Thesis-compatible connector for the complete factored Taxi state space.

The connector deliberately models perturbations of the four state factors, not
transitions in Taxi's road map.  Consequently, row/column changes cross road
walls when the corresponding factor differs by one, and states where the
passenger and destination factors are equal remain part of the 500-state
universe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from itertools import product
from typing import Final, TypeAlias

import numpy as np

from stache.explainability.core.connector import (
    ConnectorIdentity,
    DiscreteActionSpec,
    MetricCertificate,
    ObservationSpec,
)


TaxiState: TypeAlias = tuple[int, int, int, int]
TaxiKey: TypeAlias = TaxiState

_ROW_COUNT: Final = 5
_COLUMN_COUNT: Final = 5
_PASSENGER_COUNT: Final = 5
_DESTINATION_COUNT: Final = 4
_STATE_COUNT: Final = 500
_ACTION_COUNT: Final = 6

_DECLARED_STATES: Final[tuple[TaxiState, ...]] = tuple(
    product(
        range(_ROW_COUNT),
        range(_COLUMN_COUNT),
        range(_PASSENGER_COUNT),
        range(_DESTINATION_COUNT),
    )
)

_CERTIFICATE_SCOPE: Final = sha256(
    (
        "taxi-factored-500:v1|taxi-thesis-hybrid:v1|"
        "row+-1,column+-1,passenger-any-other,destination-any-other"
    ).encode("utf-8")
).hexdigest()

_ACTION_LABELS: Final[tuple[str, ...]] = (
    "south",
    "north",
    "east",
    "west",
    "pickup",
    "dropoff",
)


def _require_integer(value: object, *, factor: str) -> int:
    """Return an exact integer, rejecting bools and lossy numeric coercions."""

    if type(value) is not int:
        raise TypeError(f"Taxi {factor} must be an integer, got {value!r}")
    return value


def _validated_tuple(state: object) -> TaxiState:
    if not isinstance(state, (tuple, list)) or len(state) != 4:
        raise ValueError("Taxi state must contain exactly four integer factors")

    row = _require_integer(state[0], factor="row")
    column = _require_integer(state[1], factor="column")
    passenger = _require_integer(state[2], factor="passenger")
    destination = _require_integer(state[3], factor="destination")

    if not 0 <= row < _ROW_COUNT:
        raise ValueError(f"Taxi row {row} is outside [0, 4]")
    if not 0 <= column < _COLUMN_COUNT:
        raise ValueError(f"Taxi column {column} is outside [0, 4]")
    if not 0 <= passenger < _PASSENGER_COUNT:
        raise ValueError(f"Taxi passenger {passenger} is outside [0, 4]")
    if not 0 <= destination < _DESTINATION_COUNT:
        raise ValueError(f"Taxi destination {destination} is outside [0, 3]")

    return row, column, passenger, destination


class TaxiConnector:
    """Connector for the thesis' 500-state Taxi perturbation space."""

    identity: Final = ConnectorIdentity(
        domain="taxi",
        connector_version="1",
        state_universe="taxi-factored-500",
        state_universe_version="1",
        metric="taxi-thesis-hybrid",
        metric_version="1",
        codec="taxi-state-key",
        codec_version="1",
    )
    metric_certificate: Final = MetricCertificate(
        formal_unit=1,
        every_edge_is_formal_unit=True,
        all_valid_formal_unit_edges_present=True,
        symmetric=True,
        connected=True,
        geodesic_for_formal_metric=True,
        certificate_version="1",
        scope_fingerprint=_CERTIFICATE_SCOPE,
    )
    observation_codec_version: Final = "1"
    observation_spec: Final = ObservationSpec(shape=(_STATE_COUNT,), dtype="float32")
    action_spec: Final = DiscreteActionSpec(count=_ACTION_COUNT)

    @property
    def action_metadata(self) -> Sequence[Mapping[str, object]]:
        """Return primitive rendering-neutral metadata for Taxi's six actions."""

        return tuple(
            {"action": action, "label": label}
            for action, label in enumerate(_ACTION_LABELS)
        )

    def canonicalize(self, state: TaxiState | Sequence[int]) -> TaxiState:
        """Validate and convert a supported factored state to its tuple form."""

        return _validated_tuple(state)

    def validate_state(self, state: TaxiState | Sequence[int]) -> None:
        """Raise with a factor-specific diagnostic when ``state`` is invalid."""

        _validated_tuple(state)

    def state_key(self, state: TaxiState | Sequence[int]) -> TaxiKey:
        """Return the collision-free canonical key used by the search core."""

        return self.canonicalize(state)

    def ordering_key(self, key: TaxiKey | Sequence[int]) -> int:
        """Return a total deterministic order over canonical Taxi keys."""

        return self.encode_index(self.canonicalize(key))

    def declared_states(self) -> tuple[TaxiState, ...]:
        """Return all states, including those where passenger equals destination."""

        return _DECLARED_STATES

    def atomic_neighbors(
        self,
        state: TaxiState | Sequence[int],
    ) -> tuple[TaxiState, ...]:
        """Return every state at unit thesis distance in stable factor order."""

        row, column, passenger, destination = self.canonicalize(state)
        neighbors: list[TaxiState] = []

        for candidate_row in (row - 1, row + 1):
            if 0 <= candidate_row < _ROW_COUNT:
                neighbors.append(
                    (candidate_row, column, passenger, destination)
                )
        for candidate_column in (column - 1, column + 1):
            if 0 <= candidate_column < _COLUMN_COUNT:
                neighbors.append(
                    (row, candidate_column, passenger, destination)
                )
        neighbors.extend(
            (row, column, candidate_passenger, destination)
            for candidate_passenger in range(_PASSENGER_COUNT)
            if candidate_passenger != passenger
        )
        neighbors.extend(
            (row, column, passenger, candidate_destination)
            for candidate_destination in range(_DESTINATION_COUNT)
            if candidate_destination != destination
        )
        return tuple(neighbors)

    def formal_distance(
        self,
        left: TaxiState | Sequence[int],
        right: TaxiState | Sequence[int],
    ) -> int:
        """Compute the thesis hybrid numeric/categorical Taxi distance."""

        left_state = self.canonicalize(left)
        right_state = self.canonicalize(right)
        return (
            abs(left_state[0] - right_state[0])
            + abs(left_state[1] - right_state[1])
            + int(left_state[2] != right_state[2])
            + int(left_state[3] != right_state[3])
        )

    def formal_layers(self, seed: TaxiState | Sequence[int]) -> None:
        """Return no custom layers: the certificate proves graph geodesy."""

        self.validate_state(seed)
        return None

    def encode_index(self, state: TaxiState | Sequence[int]) -> int:
        """Encode a factored state using Taxi-v3's public 0..499 ordering."""

        row, column, passenger, destination = self.canonicalize(state)
        return (
            ((row * _COLUMN_COUNT + column) * _PASSENGER_COUNT + passenger)
            * _DESTINATION_COUNT
            + destination
        )

    def decode_index(self, index: int) -> TaxiState:
        """Decode an exact integer in 0..499 without Gymnasium."""

        if type(index) is not int:
            raise TypeError(f"Taxi index must be an integer, got {index!r}")
        if not 0 <= index < _STATE_COUNT:
            raise ValueError(f"Taxi index {index} is outside [0, 499]")

        remainder, destination = divmod(index, _DESTINATION_COUNT)
        remainder, passenger = divmod(remainder, _PASSENGER_COUNT)
        row, column = divmod(remainder, _COLUMN_COUNT)
        return row, column, passenger, destination

    def policy_lookup_key(self, state: TaxiState | Sequence[int]) -> int:
        """Return the scalar key used by precomputed Taxi policy tables."""

        return self.encode_index(state)

    def encode_observation(self, state: TaxiState | Sequence[int]) -> np.ndarray:
        """Encode a state as the flat float32 one-hot vector expected by DQN models."""

        observation = np.zeros((_STATE_COUNT,), dtype=np.float32)
        observation[self.encode_index(state)] = np.float32(1.0)
        return observation

    def encode_state(self, state: TaxiState | Sequence[int]) -> list[int]:
        """Encode a state as the codec-v1 primitive artifact representation."""

        return list(self.canonicalize(state))

    def decode_state(self, encoded: object) -> TaxiState:
        """Decode a codec-v1 state, rejecting non-list and lossy values."""

        if type(encoded) is not list:
            raise TypeError("Taxi encoded state must be a four-element list")
        return _validated_tuple(encoded)

    def encode_key(self, key: TaxiKey | Sequence[int]) -> list[int]:
        """Encode a canonical state key as a primitive four-element list."""

        return list(self.canonicalize(key))

    def decode_key(self, encoded: object) -> TaxiKey:
        """Decode a codec-v1 canonical key, rejecting lossy values."""

        if type(encoded) is not list:
            raise TypeError("Taxi encoded key must be a four-element list")
        return _validated_tuple(encoded)


__all__ = ["TaxiConnector", "TaxiKey", "TaxiState"]
