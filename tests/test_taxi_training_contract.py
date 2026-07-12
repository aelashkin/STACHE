"""Taxi training emits exactly the connector's model observation contract."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from stache.explainability.connectors.taxi import TaxiConnector
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

