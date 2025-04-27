#!/usr/bin/env python3
"""
bfs-rr.py

This module implements a BFS-based Robustness Region (BFS-RR) algorithm
for MiniGrid environments (Fetch version) in which symbolic (factored)
states are modified one atomic step at a time. Atomic steps include:
  - Changing the agent’s direction by ±1 (if within [0, 3])
  - Changing the goal’s color or type (allowed types are "key" and "ball")
  - Modifying an existing object’s attribute (color, type, state, or one coordinate of its position)
  - Creating a new object (if fewer than max_objects exist) by specifying its type, color, and position in one step

Rules include:
  1. At most a given number (max_objects) of objects may be present.
  2. Only one attribute change per step (e.g. you cannot change both color and type simultaneously).
  3. The agent’s position (if present as an object) is fixed.
  4. When comparing two states (using an L1 measure), a change in any attribute counts as “1.”
  5. In the Fetch mission space only objects of type “key” or “ball” are allowed.
  
This file uses Gymnasium 1.0.0 and assumes a Stable Baselines 3 model (or any other
agent that follows the Gymnasium API).
"""

import os
import copy
import time
import math
import datetime
from collections import deque
from PIL import Image

import yaml
import gymnasium as gym


from stache.utils.experiment_io import load_experiment
from stache.envs.minigrid_ext.environment_utils import create_minigrid_env
from stache.envs.minigrid_ext.set_state_extension import set_standard_state_minigrid, factorized_symbolic_to_fullobs
from stache.envs.minigrid_ext.constants import (
    ACTION_MAPPING_EMPTY,
    ACTION_MAPPING_FETCH,
)
from stache.envs.minigrid_ext.state_utils import (
    symbolic_to_array,
    state_to_key,
    get_grid_dimensions,
)
from stache.explainability.minigrid_neighbor_generation import get_neighbors

# variables for running main without console input

# Hardcode the model path (folder) to load the saved model.
model_path = "data/experiments/models/MiniGrid-Empty-Random-6x6-v0_PPO_model_20250304_220518"

# model_path = "data/experiments/models/MiniGrid-Fetch-5x5-N2-v0_PPO_model_20250305_031749"
max_gen_objects = 2
max_nodes_expanded = None
seed_value = 77

# === Global mappings (these must be consistent with your environment) ===

def get_symbolic_env(env):
    """
    Unwrap the environment until you find one that produces dict observations.
    This assumes that the symbolic representation is provided by a wrapper
    whose observation_space is a gym.spaces.Dict.
    """
    symbolic_env = env
    while not isinstance(symbolic_env.observation_space, gym.spaces.Dict):
        symbolic_env = symbolic_env.env
    return symbolic_env


def generate_rr_images(
    robustness_region,
    env,
    output_dir="rr_images",
    subset="all",
    subset_count=None,
):
    """
    Render and save images for states in the robustness region.

    Arguments:
        robustness_region (list): List of dicts (symbolic states).
            Each symbolic state must include the key "bfs_depth".
        env (gym.Env): An environment compatible with set_standard_state_minigrid
            (i.e., a standard MiniGrid env or a wrapper that delegates to it).
        output_dir (str): Directory in which to save the images.
        subset (str): One of {"all", "closest", "furthest"} controlling which states
            to render:
            - "all": Render all states in the RR
            - "closest": Render only states whose "bfs_depth" is minimal in the RR
            - "furthest": Render only states whose "bfs_depth" is maximal in the RR
        subset_count (int or None): If given, limit the number of states to at most
            this many (taken in BFS-discovery order).
    """

    os.makedirs(output_dir, exist_ok=True)

    if not robustness_region:
        print("No states in the robustness region; nothing to render.")
        return

    # Determine min and max BFS depth
    all_depths = [s["bfs_depth"] for s in robustness_region]
    min_depth, max_depth = min(all_depths), max(all_depths)

    # Select subset of states
    if subset == "all":
        selected_states = robustness_region
    elif subset == "closest":
        selected_states = [s for s in robustness_region if s["bfs_depth"] == min_depth]
    elif subset == "furthest":
        selected_states = [s for s in robustness_region if s["bfs_depth"] == max_depth]
    else:
        raise ValueError(f"Invalid subset option: {subset}")

    if subset_count is not None and subset_count < len(selected_states):
        selected_states = selected_states[:subset_count]

    # Render each selected state
    for i, state in enumerate(selected_states):
        depth = state["bfs_depth"]

        # Determine grid dimensions
        w, h = get_grid_dimensions(state)
        if w is None or h is None:
            w = env.width
            h = env.height

        # Convert factorized state to fullobs and apply it
        env.reset()
        full_obs = factorized_symbolic_to_fullobs(state, w, h)
        set_standard_state_minigrid(env, full_obs)

        # Render as an RGB array
        img = env.render()  # Should be a NumPy array (H x W x 3)

        # Save the image using Pillow
        filename = f"depth_{depth}_idx_{i}.png"
        filepath = os.path.join(output_dir, filename)
        Image.fromarray(img).save(filepath)

    print(f"Saved {len(selected_states)} images to '{output_dir}'.")


# === BFS-RR Algorithm ===

def bfs_rr(
    initial_state,
    model,
    env_name,
    max_obs_objects=10,  # For symbolic_to_array / PaddedObservation
    max_walls=25,
    max_gen_objects=2,   # For neighbor generation only
    max_nodes_expanded=100,       # Maximum number of nodes to dequeue/expand
):
    """
    Computes the Robustness Region for the given initial_state under the policy
    represented by model. The region is defined as the set of all symbolic states
    for which the model produces the same action as for the initial_state.

    Parameters:
      - initial_state (dict): a dictionary representing the factored state.
      - model: a Stable Baselines 3 model (or similar) that has a predict(obs, deterministic=True)
               method. The observation must be produced via symbolic_to_array().
      - env_name (str): The name of the MiniGrid environment (e.g., 'MiniGrid-Fetch-5x5-N2-v0').
      - max_obs_objects (int): maximum number of objects used for the observation's fixed-size array.
      - max_walls (int): maximum number of outer walls (used for observation conversion).
      - max_gen_objects (int): maximum number of objects allowed in the state when generating neighbors
                               (i.e., the BFS expansions).
      - max_nodes_expanded (int): maximum number of nodes to dequeue/expand in the BFS.

    Returns:
      - robustness_region (list): a list of symbolic states (each a dict) in the robustness region,
                                  in the order discovered, with an extra key "bfs_depth" storing the 
                                  L1 distance from the initial state.
      - stats (dict): dictionary containing statistics about the BFS, including:
          * "initial_action"
          * "region_size": number of states in the robustness region
          * "total_opened_nodes": total number of nodes expanded
          * "visited_count": total number of visited states
          * "elapsed_time": time taken (seconds)
    """

    start_time = time.time()
    total_opened_nodes = 0

    if max_nodes_expanded is None:
        max_nodes_expanded = math.inf

    # 1. Predict initial action
    initial_obs = symbolic_to_array(initial_state, max_obs_objects, max_walls)
    initial_action, _ = model.predict(initial_obs, deterministic=True)

    # 2. Initialize BFS data structures
    region = {}            # Maps state_key -> the actual state (with "bfs_depth")
    visited = set()        # set of state_keys
    queue = deque()

    # 3. Enqueue the initial state with depth=0
    init_key = state_to_key(initial_state)
    visited.add(init_key)
    queue.append((initial_state, 0))

    robustness_region = []  # final list of states in the region

    # 4. BFS
    while queue and total_opened_nodes < max_nodes_expanded:
        total_opened_nodes += 1

        state, depth = queue.popleft()
        key = state_to_key(state)

        # Predict the action for this state
        obs = symbolic_to_array(state, max_obs_objects, max_walls)
        action, _ = model.predict(obs, deterministic=True)

        if action == initial_action:
            # If not yet in region, store it with BFS depth
            if key not in region:
                state_copy = copy.deepcopy(state)
                state_copy["bfs_depth"] = depth
                region[key] = state_copy
                robustness_region.append(state_copy)

            # Expand neighbors
            neighbors = get_neighbors(state, env_name=env_name, max_gen_objects=max_gen_objects)
            for neighbor in neighbors:
                nkey = state_to_key(neighbor)
                if nkey not in visited:
                    visited.add(nkey)
                    queue.append((neighbor, depth + 1))

    elapsed_time = time.time() - start_time
    stats = {
        "initial_action": initial_action,
        "region_size": len(region),
        "total_opened_nodes": total_opened_nodes,
        "visited_count": len(visited),
        "elapsed_time": elapsed_time,
    }

    return robustness_region, stats




# === Example Usage ===
if __name__ == '__main__':

    # Currently hardcoded at the top of the file
    # Hardcode the model path (folder) to load the saved model.
    # model_path = "data/experiments/models/MiniGrid-Fetch-5x5-N2-v0_PPO_model_20250305_031749"

    model, config_data = load_experiment(model_path)
    env_config = config_data["env_config"]  # Extract the environment configuration

    # Set render_mode to human for an on-screen render (optional).
    env_config["render_mode"] = "human"

    print(f"Loaded model for environment: {env_config['env_name']}")
    print(f"Configuration: {env_config}")

    # --- Create the symbolic MiniGrid environment ---
    env = create_minigrid_env(env_config)
    symbolic_env = get_symbolic_env(env)

    # Use a fixed seed value (as defined earlier).
    # seed_value = 42

    # Reset the environment to get the initial symbolic state.
    initial_symbolic_state, info = symbolic_env.reset(seed=seed_value, options={})
    env.render()

    # Reset the full environment.
    initial_state, info = env.reset(seed=seed_value, options={})
    env.render()

    action, _ = model.predict(initial_state, deterministic=True)
    print(f"Initial action: {action}")

    # Print action mapping table based on the environment name
    env_name_lower = env_config['env_name'].lower()
    if "fetch" in env_name_lower:
        print("Action space mapping for 'fetch': \n (Num, Name, Action)")
        for row in ACTION_MAPPING_FETCH:
            print(row)
    elif "empty" in env_name_lower:
        print("Action space mapping for 'empty': (Num, Name, Action)")
        for row in ACTION_MAPPING_EMPTY:
            print(row)

    print(f"Initial symbolic state: {initial_symbolic_state}")
    print(f"Initial state: {initial_state}")

    # --- Calculate the Robustness Region ---
    robustness_region, stats = bfs_rr(
        initial_symbolic_state,
        model,
        env_name=env_config["env_name"],
        max_obs_objects=env_config["max_objects"],
        max_walls=env_config["max_walls"],
        max_gen_objects=max_gen_objects,
        max_nodes_expanded=max_nodes_expanded
    )

    # Print useful statistics
    print("\nRobustness Region Statistics:")
    print(f"Inital action: {stats['initial_action']}")
    print(f"Region size: {stats['region_size']}")
    print(f"Total opened nodes: {stats['total_opened_nodes']}")
    print(f"Visited nodes count: {stats['visited_count']}")
    print(f"Elapsed time: {stats['elapsed_time']:.2f} seconds")
    print("\nFirst 10 states in the robustness region:")
    for state in robustness_region[:10]:
        print(state_to_key(state))

    # Prepare metadata for YAML file
    model_name = os.path.basename(model_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rr_dir = os.path.join("data", "experiments", "rr", model_name, f"seed_{seed_value}_time_{timestamp}")
    os.makedirs(rr_dir, exist_ok=True)
    yaml_file_path = os.path.join(rr_dir, "robustness_region.yaml")

    metadata = {
        "metadata": {
            "model_name": model_name,
            "seed": seed_value,
            "timestamp": timestamp,
            "region_size": stats["region_size"],
            "total_opened_nodes": stats["total_opened_nodes"],
            "visited_count": stats["visited_count"],
            "elapsed_time": stats["elapsed_time"],
            "env_config": env_config,
        },
        "robustness_region": robustness_region,
    }

    with open(yaml_file_path, "w") as f:
        yaml.dump(metadata, f)

    print(f"\nRobustness region and metadata saved to: {yaml_file_path}")

    env_config["render_mode"] = "rgb_array"
    render_env = create_minigrid_env(env_config)

    # --- Render and save images for all states in the robustness region ---
    images_output_dir = os.path.join(rr_dir, "images")
    generate_rr_images(
        robustness_region=robustness_region,
        env=render_env,
        output_dir=images_output_dir,
        subset="all",            # Render all states
        subset_count=None        # No limit to how many we save
    )
    print(f"Rendered images for the entire RR to: {images_output_dir}")

    env.close()
