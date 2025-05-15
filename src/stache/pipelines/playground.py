from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import COLOR_NAMES

import torch as th
import torch.nn as nn

import argparse
import os
import multiprocessing as mp

class CustomMiniGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            dummy_obs_sample = observation_space.sample()
            dummy_input = th.as_tensor(dummy_obs_sample[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        cnn_features = self.cnn(observations)
        linear_output = self.linear(cnn_features)
        return linear_output

class OneHotEncoder(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: th.Tensor) -> th.Tensor:
        if x.ndim == 2 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if not (x.dtype == th.int64 or x.dtype == th.int32):
            x = x.long()
        one_hot_output = th.nn.functional.one_hot(x, num_classes=self.num_classes).float()
        return one_hot_output

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, cnn_features_dim: int = 64, mission_embedding_dim: int = 16):
        super().__init__(observation_space, features_dim=1)  # Placeholder, self._features_dim is set below
        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                extractors[key] = CustomMiniGridCNN(subspace, features_dim=cnn_features_dim)
                total_concat_size += cnn_features_dim
            elif key == "mission":
                if not isinstance(subspace, spaces.Discrete):
                    raise ValueError(f"Mission subspace '{key}' must be of type spaces.Discrete for nn.Embedding, got {type(subspace)}")
                extractors[key] = nn.Embedding(num_embeddings=subspace.n, embedding_dim=mission_embedding_dim)
                total_concat_size += mission_embedding_dim
            elif isinstance(subspace, spaces.Discrete):
                extractors[key] = nn.Identity()
                total_concat_size += subspace.n
            elif isinstance(subspace, spaces.Box) and len(subspace.shape) == 1:
                extractors[key] = nn.Flatten()
                total_concat_size += int(np.prod(subspace.shape))
            else:
                raise ValueError(f"Unsupported subspace type for key '{key}': {subspace} with shape {subspace.shape}")
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: PyTorchObs) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            observation_data = observations[key]
            processed_observation_data = observation_data
            if key == "mission":
                processed_observation_data = th.argmax(observation_data, dim=-1)
            elif key == "image" and processed_observation_data.dtype != th.float32:
                processed_observation_data = processed_observation_data.float()
            extracted_features = extractor(processed_observation_data)
            if extracted_features.ndim == 3 and extracted_features.shape[1] == 1:
                extracted_features = extracted_features.squeeze(1)
            if extracted_features.ndim != 2:
                raise ValueError(
                    f"Extractor for key '{key}' ({type(extractor).__name__}) produced features with ndim={extracted_features.ndim} "
                    f"(shape {extracted_features.shape}), after processing. Expected 2D."
                )
            encoded_tensor_list.append(extracted_features)
        if not encoded_tensor_list:
            print("WARNING: encoded_tensor_list is empty!")
            batch_size_if_known = 0
            if observations and isinstance(observations, dict) and observations.keys():
                first_key = list(observations.keys())[0]
                if hasattr(observations[first_key], 'shape') and len(observations[first_key].shape) > 0:
                    batch_size_if_known = observations[first_key].shape[0]
            device_to_use = 'cpu'
            if self.extractors:
                first_extractor_key = list(self.extractors.keys())[0]
                try:
                    device_to_use = next(self.extractors[first_extractor_key].parameters()).device
                except StopIteration:
                    pass
            return th.empty((batch_size_if_known, 0), device=device_to_use)
        concatenated_output = th.cat(encoded_tensor_list, dim=1)
        return concatenated_output

class MissionStringEncoderWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.mission_syntaxes = [
            "get a", "go get a", "fetch a", "go fetch a", "you must fetch a"
        ]
        self.mission_colors = COLOR_NAMES
        self.mission_types = ["key", "ball"]
        self.possible_missions = []
        for syntax in self.mission_syntaxes:
            for color in self.mission_colors:
                for obj_type in self.mission_types:
                    self.possible_missions.append(f"{syntax} {color} {obj_type}")
        self.mission_to_idx = {mission: i for i, mission in enumerate(self.possible_missions)}
        self.num_possible_missions = len(self.possible_missions)
        original_obs_space_dict = env.observation_space.spaces
        new_mission_space = spaces.Discrete(self.num_possible_missions)
        self.observation_space = spaces.Dict({**original_obs_space_dict, 'mission': new_mission_space})

    def observation(self, obs):
        mission_str = obs['mission']
        if mission_str not in self.mission_to_idx:
            print(f"Warning: Unknown mission string '{mission_str}' encountered in MissionStringEncoderWrapper. Defaulting to index 0.")
            obs['mission'] = 0
        else:
            obs['mission'] = self.mission_to_idx[mission_str]
        return obs

def create_env_fn(env_name="MiniGrid-Fetch-5x5-N2-v0", render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    env = FullyObsWrapper(env)
    env = MissionStringEncoderWrapper(env)
    return env

def run(device: str, steps: int):
    env_id = "MiniGrid-Fetch-5x5-N2-v0"
    N_ENVS = 8
    vec_env = make_vec_env(lambda: create_env_fn(env_id), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    ppo_hyperparams = {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.015,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
    }
    policy_kwargs_ppo = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(
            cnn_features_dim=64,
            mission_embedding_dim=16
        ),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs_ppo,
        device=device,
        verbose=1,
        **ppo_hyperparams
    )
    print(f"Starting training PPO with MultiInputPolicy on {env_id}...")
    model.learn(total_timesteps=steps)
    eval_env = create_env_fn(env_id)
    n_eval_episodes = 100
    print("Evaluating the agent on training environment...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Evaluation results: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    eval_env.close()

    # model.save("ppo_minigrid_fetch5x5_multi_input_embedded_mission") 
    # print("Training complete. Model saved.")
    
    vec_env.close()
    del model, vec_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--steps", type=int, default=200_000)
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    run(args.device, args.steps)
