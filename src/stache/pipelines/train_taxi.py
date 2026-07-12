"""Train and persist the Taxi-v3 DQN used by the RR workflows.

Importing this module is side-effect free.  Training is intentionally available
only through :func:`main` so test collection and package inspection cannot start
a long-running job.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from stache.explainability.connectors.taxi import TaxiConnector
from stache.utils.experiment_io import save_experiment


class OneHotObs(gym.ObservationWrapper):
    """Expose exactly the observation identity declared by ``TaxiConnector``."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        connector = TaxiConnector()
        if not isinstance(env.observation_space, spaces.Discrete):
            raise TypeError("OneHotObs only supports a discrete observation space")
        if env.observation_space.n != len(connector.declared_states()):
            raise ValueError(
                "Taxi training requires the connector's 500-state universe"
            )
        if (
            not isinstance(env.action_space, spaces.Discrete)
            or env.action_space.n != connector.action_spec.count
        ):
            raise ValueError(
                "Taxi training action space does not match the connector contract"
            )
        self._connector = connector
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=connector.observation_spec.shape,
            dtype=np.dtype(connector.observation_spec.dtype),
        )

    def observation(self, observation: object) -> np.ndarray:
        if isinstance(observation, np.integer):
            observation = int(observation)
        state = self._connector.decode_index(observation)
        return self._connector.encode_observation(state)


def make_env() -> OneHotObs:
    """Create one Taxi-v3 environment with connector-owned observations."""

    return OneHotObs(gym.make("Taxi-v3"))


def train_and_save() -> dict[str, str]:
    """Train, evaluate, and save a model plus its semantic manifest."""

    connector = TaxiConnector()
    n_envs = 8
    train_env = DummyVecEnv([make_env for _ in range(n_envs)])
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        buffer_size=500_000,
        learning_starts=10_000,
        batch_size=128,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_initial_eps=1.0,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=42,
    )

    total_timesteps_to_train = 600_000
    model.learn(total_timesteps=total_timesteps_to_train)

    eval_env = Monitor(make_env())
    n_eval_episodes = 100
    print("Evaluating the agent on training environment...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    env_config = {
        "env_name": "Taxi-v3",
        "n_envs": n_envs,
        "wrapper": "OneHotObs",
    }
    model_config = {
        "model_type": "DQN",
        "policy": "MlpPolicy",
        "learning_rate": 1e-4,
        "buffer_size": 500_000,
        "learning_starts": 10_000,
        "batch_size": 128,
        "gamma": 0.99,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "target_update_interval": 10_000,
        "exploration_initial_eps": 1.0,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.02,
        "policy_kwargs": {"net_arch": [256, 256]},
        "seed": 42,
        "total_timesteps": total_timesteps_to_train,
    }
    training_log = (
        f"Training completed for {total_timesteps_to_train} timesteps.\n"
        f"Evaluation over {n_eval_episodes} episodes:\n"
        f"Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}\n"
    )

    print("\nSaving the experiment...")
    experiment_info = save_experiment(
        model=model,
        env_config=env_config,
        model_config=model_config,
        training_log=training_log,
        model_connector=connector,
    )
    print("Experiment saved successfully:")
    print(experiment_info)
    return experiment_info


def main() -> int:
    train_and_save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
