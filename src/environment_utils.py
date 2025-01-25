import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from minigrid.wrappers import FullyObsWrapper
from src.wrappers import FactorizedSymbolicWrapper, PaddedObservationWrapper


def create_symbolic_minigrid_env(env_config: dict) -> gym.Env:
    """
    Create and set up the MiniGrid environment.
    """
    env_name = env_config["env_name"]
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = FactorizedSymbolicWrapper(env)
    env = PaddedObservationWrapper(env, max_objects=env_config["max_objects"], max_walls=env_config["max_walls"])
    env = Monitor(env)
    return env