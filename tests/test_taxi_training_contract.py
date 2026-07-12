"""Taxi training emits exactly the connector's model observation contract."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from stache.explainability.connectors.taxi import TaxiConnector
from stache.pipelines import train_taxi
from stache.pipelines.train_taxi import OneHotObs


def test_training_wrapper_matches_connector_for_all_500_states() -> None:
    connector = TaxiConnector()
    wrapper = OneHotObs(gym.make("Taxi-v3"))
    try:
        for index in range(500):
            np.testing.assert_array_equal(
                wrapper.observation(index),
                connector.encode_observation(connector.decode_index(index)),
            )
    finally:
        wrapper.close()


def test_training_help_never_starts_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = False

    def unexpected_training() -> dict[str, str]:
        nonlocal started
        started = True
        return {}

    monkeypatch.setattr(train_taxi, "train_and_save", unexpected_training)

    with pytest.raises(SystemExit) as exit_info:
        train_taxi.main(["--help"])

    assert exit_info.value.code == 0
    assert started is False
