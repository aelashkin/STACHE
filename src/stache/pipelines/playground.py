from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs # <<< MOVED HERE

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


# This is the CustomMiniGridCNN from your original script, now with added print statements
class CustomMiniGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        # print(f"  [CustomMiniGridCNN.__init__] observation_space shape: {observation_space.shape}, features_dim: {features_dim}")
        
        # Assuming observation_space.shape is (C, H, W) due to VecTransposeImage
        n_input_channels = observation_space.shape[0]
        # print(f"    n_input_channels: {n_input_channels}")
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),
            nn.Flatten(), # Default start_dim=1, flattens (B, C_out, H_out, W_out) to (B, C_out*H_out*W_out)
        )

        with th.no_grad():
            dummy_obs_sample = observation_space.sample() # Expected (C, H, W)
            dummy_input = th.as_tensor(dummy_obs_sample[None]).float() # Adds batch dim -> (1, C, H, W)
            # print(f"    CustomMiniGridCNN dummy_input for n_flatten calc: shape {dummy_input.shape}")
            
            # For detailed debugging of CNN structure:
            # temp_out_cnn_init = dummy_input
            # for i, layer in enumerate(self.cnn):
            #     temp_out_cnn_init = layer(temp_out_cnn_init)
            #     print(f"    CustomMiniGridCNN __init__ dummy_input after layer {i} ({type(layer).__name__}): shape {temp_out_cnn_init.shape}")
            
            n_flatten = self.cnn(dummy_input).shape[1] # Output of cnn is (1, n_flatten)
            # print(f"    Calculated n_flatten: {n_flatten}")

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        # print(f"  [CustomMiniGridCNN.__init__] CNN module: {self.cnn}")
        # print(f"  [CustomMiniGridCNN.__init__] Linear module: {self.linear}")

    def forward(self, observations: th.Tensor) -> th.Tensor:
        print(f"    [CustomMiniGridCNN.forward] Input 'observations' (image): shape {observations.shape}, dtype {observations.dtype}")
        
        # For detailed debugging of CNN forward pass:
        # temp_out_cnn_fwd = observations
        # for i, layer in enumerate(self.cnn):
        #     # prev_shape = temp_out_cnn_fwd.shape
        #     temp_out_cnn_fwd = layer(temp_out_cnn_fwd)
        #     print(f"      CNN forward layer {i} ({type(layer).__name__}) output shape: {temp_out_cnn_fwd.shape}")
        
        cnn_features = self.cnn(observations) # Expected output: (batch_size, n_flatten)
        print(f"    [CustomMiniGridCNN.forward] Output of self.cnn (after flatten): shape {cnn_features.shape}, dtype {cnn_features.dtype}")
        
        linear_output = self.linear(cnn_features) # Expected output: (batch_size, features_dim)
        print(f"    [CustomMiniGridCNN.forward] Output of self.linear: shape {linear_output.shape}, dtype {linear_output.dtype}")
        return linear_output

# This is the OneHotEncoder from your previous version, with added print statements
class OneHotEncoder(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: th.Tensor) -> th.Tensor:
        # print(f"      [OneHotEncoder.forward] Input 'x': shape {x.shape}, dtype {x.dtype}")
        original_shape = x.shape
        if x.ndim == 2 and x.shape[-1] == 1:
            x = x.squeeze(-1)
            # print(f"        OneHotEncoder after squeeze: shape {x.shape}")
        
        if not (x.dtype == th.int64 or x.dtype == th.int32):
             x = x.long()
            # print(f"        OneHotEncoder after .long(): dtype {x.dtype}")
        
        one_hot_output = th.nn.functional.one_hot(x, num_classes=self.num_classes).float()
        # print(f"      [OneHotEncoder.forward] Output: shape {one_hot_output.shape}, dtype {one_hot_output.dtype}")
        return one_hot_output

# This is the CustomCombinedExtractor from your previous version, with added print statements
# MODIFIED CustomCombinedExtractor
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, cnn_features_dim: int = 64, mission_embedding_dim: int = 16):
        super().__init__(observation_space, features_dim=1) # Placeholder, self._features_dim is set below
        # print(f"[CustomCombinedExtractor.__init__] Initializing...")
        # print(f"  Input observation_space: {observation_space}")

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            # print(f"  Processing key: '{key}', subspace: {subspace}")
            if key == "image":
                # print(f"    Initializing CustomMiniGridCNN for '{key}' with features_dim={cnn_features_dim}")
                extractors[key] = CustomMiniGridCNN(subspace, features_dim=cnn_features_dim)
                total_concat_size += cnn_features_dim
            elif key == "mission":
                if not isinstance(subspace, spaces.Discrete):
                    raise ValueError(f"Mission subspace '{key}' must be of type spaces.Discrete for nn.Embedding, got {type(subspace)}")
                # print(f"    Initializing nn.Embedding for '{key}' with num_embeddings={subspace.n}, embedding_dim={mission_embedding_dim}")
                extractors[key] = nn.Embedding(num_embeddings=subspace.n, embedding_dim=mission_embedding_dim)
                total_concat_size += mission_embedding_dim
            elif isinstance(subspace, spaces.Discrete): 
                # print(f"    Initializing nn.Identity for discrete key '{key}' (already one-hot encoded by SB3) with size={subspace.n}")
                # SB3 automatically one-hot encodes Discrete spaces in a Dict observation.
                # So, the input for 'direction' will be (batch_size, subspace.n).
                # nn.Identity() will pass it through as a 2D tensor.
                extractors[key] = nn.Identity() # Handles 'direction' (which is auto-one-hot-encoded by SB3)
                total_concat_size += subspace.n
            elif isinstance(subspace, spaces.Box) and len(subspace.shape) == 1: 
                # print(f"    Initializing nn.Flatten for 1D Box key '{key}' with shape {subspace.shape}")
                extractors[key] = nn.Flatten()
                total_concat_size += int(np.prod(subspace.shape))
            else:
                raise ValueError(f"Unsupported subspace type for key '{key}': {subspace} with shape {subspace.shape}")
            # print(f"    Current total_concat_size: {total_concat_size}")

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
        # print(f"[CustomCombinedExtractor.__init__] Final _features_dim: {self._features_dim}")
        # print(f"  Extractors dict: {self.extractors}")

    def forward(self, observations: PyTorchObs) -> th.Tensor:
        print("\n[CustomCombinedExtractor.forward] Called.")
        # print(f"  Input observations type: {type(observations)}")
        # for key, val_tensor in observations.items():
        #     print(f"  observations['{key}']: shape {val_tensor.shape}, dtype {val_tensor.dtype}, device {val_tensor.device}")

        encoded_tensor_list = []
        print("  Iterating through extractors to process observation components:")
        for key, extractor in self.extractors.items():
            print(f"  Processing observation component: '{key}' using {type(extractor).__name__}")
            observation_data = observations[key]
            print(f"    Input data for '{key}': shape {observation_data.shape}, dtype {observation_data.dtype}")

            processed_observation_data = observation_data # Initialize with current observation component

            if key == "mission":
                # 'observation_data' for mission is one-hot encoded by SB3: (batch_size, num_mission_classes), float32
                # nn.Embedding expects indices: (batch_size,), int64
                # Convert one-hot tensor to indices tensor.
                processed_observation_data = th.argmax(observation_data, dim=1)
                # th.argmax returns int64, which is the required dtype for nn.Embedding indices.
                print(f"      '{key}' (mission) after argmax to get indices: shape {processed_observation_data.shape}, dtype {processed_observation_data.dtype}")
            
            elif key == "image":
                # Input for CustomMiniGridCNN
                # VecTransposeImage wrapper ensures CHW format. Conv2D expects float.
                if processed_observation_data.dtype != th.float32:
                    # print(f"      '{key}' (image) is {processed_observation_data.dtype}, converting to float32 for CNN.")
                    # Assuming pixel values are 0-255 if uint8, else scale if needed or just convert type.
                    # MiniGrid's FullyObsWrapper provides image values already in a range (e.g. 0-10), not 0-255.
                    # The original code had / 255.0 conditional on uint8. MiniGrid image obs are often int.
                    # For safety, ensure it's float. Scaling depends on expected range by CNN.
                    # PPO's default CNN preprocesses by /255.0. Here, this is custom.
                    # The provided code did not normalize for FullyObsWrapper output (which are not pixels 0-255).
                    # Let's stick to just converting to float if not already, assuming CustomMiniGridCNN handles raw values.
                    processed_observation_data = processed_observation_data.float()
            
            # For 'direction', if using nn.Identity(), 'processed_observation_data' (which is 'observation_data')
            # is already one-hot encoded and float32, ready for nn.Identity.
            
            extracted_features = extractor(processed_observation_data)
            print(f"    Output features from '{key}' extractor: shape {extracted_features.shape}, dtype {extracted_features.dtype}")
            encoded_tensor_list.append(extracted_features)

        print("  --- Tensor details before concatenation ---")
        if not encoded_tensor_list:
            print("  WARNING: encoded_tensor_list is empty!")
            batch_size_if_known = observations[list(observations.keys())[0]].shape[0] if observations else 0
            return th.empty((batch_size_if_known, 0), device=self.extractors[list(self.extractors.keys())[0]].linear[0].weight.device if self.extractors and hasattr(self.extractors[list(self.extractors.keys())[0]], 'linear') else 'cpu')


        for i, t in enumerate(encoded_tensor_list):
            key_name = list(self.extractors.keys())[i]
            print(f"  Tensor for '{key_name}': shape {t.shape}, ndim {t.ndim}, dtype {t.dtype}")

        print(f"  Attempting to concatenate {len(encoded_tensor_list)} tensors along dim=1.")
        try:
            concatenated_output = th.cat(encoded_tensor_list, dim=1)
        except RuntimeError as e:
            print(f"  !!! RuntimeError during th.cat: {e} !!!")
            print(f"  Tensor details at point of th.cat failure:")
            for j, tensor_in_list in enumerate(encoded_tensor_list):
                k_name = list(self.extractors.keys())[j]
                print(f"    {j}. '{k_name}': shape {tensor_in_list.shape}, ndim {tensor_in_list.ndim}, dtype {tensor_in_list.dtype}")
            raise e 
            
        print(f"  Output of th.cat: shape {concatenated_output.shape}, dtype {concatenated_output.dtype}")
        print("[CustomCombinedExtractor.forward] Finished successfully.")
        return concatenated_output

# Custom Wrapper to Encode Mission Strings (from previous step)
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
        if 'mission' not in original_obs_space_dict:
            pass

        new_mission_space = spaces.Discrete(self.num_possible_missions)
        
        self.observation_space = spaces.Dict({
            **original_obs_space_dict,
            'mission': new_mission_space
        })

    def observation(self, obs):
        mission_str = obs['mission']
        if mission_str not in self.mission_to_idx:
            print(f"Warning: Unknown mission string '{mission_str}' encountered. Defaulting to index 0.")
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
    parser.add_argument("--steps", type=int, default=20_000) 
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True) 
    run(args.device, args.steps)