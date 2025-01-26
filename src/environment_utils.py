import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from minigrid.wrappers import FullyObsWrapper
from src.wrappers import FactorizedSymbolicWrapper, PaddedObservationWrapper

# from stable_baselines3.common.preprocessing import BaseFeaturesExtractor
# import torch
# import torch.nn as nn

def create_symbolic_minigrid_env(env_config: dict) -> gym.Env:
    """
    Create and set up the MiniGrid environment.
    """
    env_name = env_config.get("env_name")
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = FactorizedSymbolicWrapper(env)
    env = PaddedObservationWrapper(env, max_objects=env_config["max_objects"], max_walls=env_config["max_walls"])
    env = Monitor(env)
    return env


def create_standard_minigrid_env(env_config: dict) -> gym.Env:
    """
    Create and set up the MiniGrid environment without symbolic observation.
    """
    env_name = env_config["env_name"]
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = Monitor(env)
    return env

def create_image_minigrid_env(env_config: dict) -> gym.Env:
    pass

def create_minigrid_env(env_config: dict) -> gym.Env:
    env_name = env_config["env_name"]
    print(f"Initializing the environment: {env_name}")

    representation = env_config.get("representation")

    if representation == "symbolic":
        env = create_symbolic_minigrid_env(env_config)
        print("Using symbolic representation.")
        return env
    elif representation == "image":
        env = create_image_minigrid_env(env_config)
        print("Using image representation.")
        return env
    elif representation == "standard":
        env = create_standard_minigrid_env(env_config)
        print("Using standard representation.")
        return env
    else:
        raise ValueError(f"Unknown representation: {representation}. Valid options are: 'symbolic', 'image', 'standard'.")




# class MinigridFeaturesExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
#         super().__init__(observation_space, features_dim)
#         n_input_channels = observation_space.shape[0]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 16, (2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, (2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, (2, 2)),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Compute shape by doing one forward pass
#         with torch.no_grad():
#             n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         return self.linear(self.cnn(observations))