import torch
import torch.nn as nn

import gymnasium as gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper, ActionBonus, PositionBonus

# ────────────────────────────────────────────────────────────────────────────────
# MiniGrid environment factory
# ────────────────────────────────────────────────────────────────────────────────

def create_minigrid_env(env_config: dict) -> gym.Env:
    env_name = env_config["env_name"]
    print(f"Initializing the environment: {env_name}")

    representation = env_config.get("representation")

    if representation == "symbolic":
        env = create_symbolic_minigrid_env(env_config)
        print("Using symbolic representation.")
    elif representation == "image":
        env = create_image_minigrid_env(env_config)
        print("Using image representation.")
    elif representation == "standard":
        env = create_standard_minigrid_env(env_config)
        print("Using standard representation.")
    else:
        raise ValueError(f"Unknown representation: {representation}. Valid options are: 'symbolic', 'image', 'standard'.")
    
    # Apply reward wrappers based on configuration
    wrapper_config = env_config.get("reward_wrappers", {})
    env = apply_reward_wrappers(env, wrapper_config)

    env = Monitor(env)

    return env

def create_image_minigrid_env(env_config: dict) -> gym.Env:
    raise NotImplementedError("Image-based observation is not yet supported.")

def create_standard_minigrid_env(env_config: dict) -> gym.Env:
    """
    Create and set up the MiniGrid environment without symbolic observation.
    """
    env_name = env_config.get("env_name")
    render_mode = env_config.get("render_mode")
    env = gym.make(env_name, render_mode=render_mode)
    env = FullyObsWrapper(env)
    # env = ImgObsWrapper(env)
    env = FlatObsWrapper(env)
        
    if env is None:
        print(f"Failed to create the environment: {env_name}")
    return env

def create_symbolic_minigrid_env(env_config: dict) -> gym.Env:
    """
    Create and set up the MiniGrid environment.
    """
    env_name = env_config.get("env_name")
    render_mode = env_config.get("render_mode")
    #TODO: fix this in future implementation, so proper config is saved and passed
    include_walls = env_config.get("include_walls", True)
    env = gym.make(env_name, render_mode=render_mode)
    env = FullyObsWrapper(env)
    env = FactorizedSymbolicWrapper(env)
    
    env = PaddedObservationWrapper(env, 
                                   max_objects=env_config["max_objects"], 
                                   max_walls=env_config["max_walls"], 
                                   include_walls=include_walls)

    return env


# ────────────────────────────────────────────────────────────────────────────────
# Reward wrappers section
# ────────────────────────────────────────────────────────────────────────────────

def apply_reward_wrappers(env, wrapper_config):
    """
    Apply reward wrappers based on configuration.
    
    Args:
        env: The environment to wrap
        wrapper_config: Dictionary with wrapper configuration
        
    Returns:
        The wrapped environment
    """
    if wrapper_config is None:
        return env
        
    # Apply ActionBonus wrapper if enabled
    if wrapper_config.get("action_bonus", True):
        env = ActionBonus(env)
        print("Applied ActionBonus wrapper.")
        
    # Apply PositionBonus wrapper if enabled
    if wrapper_config.get("position_bonus", True):
        env = PositionBonus(env)
        print("Applied PositionBonus wrapper.")

    return env



class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))