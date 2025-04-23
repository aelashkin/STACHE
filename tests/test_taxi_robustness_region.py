import pytest
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN

from explainability.robust_taxi import (
    translate_tuple_to_onehot,
    get_neighbors_taxi,
    compute_rr_taxi,
)

class DummyModelAlwaysZero:
    """Mock model that always predicts action 0"""
    def predict(self, obs, deterministic=True):
        # obs shape (1,500)
        return 0, None

class DummyModelIncremental:
    """Mock model that predicts action equal to sum of one-hot index mod 6"""
    def predict(self, obs, deterministic=True):
        vec = obs[0]
        idx = int(np.argmax(vec))
        return idx % 6, None


def test_get_neighbors_taxi_count_and_uniqueness():
    f = (2, 2, 0, 1)
    neighbors = get_neighbors_taxi(f)
    # 2 row moves + 2 col moves + 4 passenger changes + 3 destination changes = 11
    assert len(neighbors) == 11
    # All neighbors unique and differ by exactly one component
    assert len(set(neighbors)) == len(neighbors)
    diffs = []
    for n in neighbors:
        diff = sum(1 for a, b in zip(f, n) if a != b)
        assert diff == 1


def test_translate_tuple_to_onehot_and_roundtrip():
    env = gym.make('Taxi-v3')
    for f in [(0,0,0,0), (4,4,4,3), (1,3,2,0)]:
        vec = translate_tuple_to_onehot(env, f)
        assert vec.shape == (1, env.observation_space.n)
        idx = int(np.argmax(vec[0]))
        # unwrap TimeLimit wrapper to access decode
        base_env = getattr(env, 'unwrapped', env)
        f2 = base_env.decode(idx)  # type: ignore[attr-defined]
        # decode may return list or named tuple, convert to tuple
        f2_tup = tuple(f2)
        assert f2_tup == f

    # Test invalid ranges
    with pytest.raises(ValueError):
        translate_tuple_to_onehot(env, (5,0,0,0))
    with pytest.raises(ValueError):
        translate_tuple_to_onehot(env, (0,6,0,0))
    with pytest.raises(ValueError):
        translate_tuple_to_onehot(env, (0,0,5,0))
    with pytest.raises(ValueError):
        translate_tuple_to_onehot(env, (0,0,0,4))


def test_compute_rr_taxi_full_coverage_for_constant_model():
    env = gym.make('Taxi-v3')
    model = DummyModelAlwaysZero()
    seed = (0, 0, 4, 0)
    result = compute_rr_taxi(seed, model, env)
    rr_set = result['rr_tuple_set']
    depths = result['rr_depths']
    stats = result['stats']
    # Constant model yields full connectivity: all 500 states
    assert len(rr_set) == env.observation_space.n
    assert stats['region_size'] == env.observation_space.n
    # Depths should be assigned for all states
    assert set(depths.keys()) == rr_set
    # visited and opened at least number of states
    assert stats['visited'] >= len(rr_set)
    assert stats['opened'] >= len(rr_set)


def test_compute_rr_taxi_varying_model_limits_region():
    env = gym.make('Taxi-v3')
    # dummy model that yields different actions => only seed remains
    model = DummyModelIncremental()
    seed = (0, 0, 4, 0)
    # precomputed_sa mapping to force only seed matches
    # map seed state index to action seed_action, others to other action
    base_env = getattr(env, 'unwrapped', env)
    seed_idx = base_env.encode(*seed)  # type: ignore[attr-defined]
    seed_action, _ = model.predict(translate_tuple_to_onehot(env, seed), deterministic=True)
    precomputed = {i: (seed_action if i == seed_idx else seed_action + 1) for i in range(env.observation_space.n)}
    result = compute_rr_taxi(seed, model, env, precomputed_sa=precomputed)
    rr_set = result['rr_tuple_set']
    stats = result['stats']
    assert rr_set == {seed}
    assert stats['region_size'] == 1

if __name__ == '__main__':
    pytest.main(['-q', __file__])
