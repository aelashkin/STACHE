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

import copy
from collections import deque

import gymnasium as gym
import numpy as np


from utils import load_experiment
from environment_utils import create_minigrid_env


# variables for running main without console input

# Hardcode the model path (folder) to load the saved model.
model_path = "data/experiments/models/MiniGrid-Fetch-5x5-N2-v0_PPO_model_20250305_031749"
max_gen_objects = 2

# === Global mappings (these must be consistent with your environment) ===

COLOR_TO_IDX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "purple": 3,
    "yellow": 4,
    "grey": 5,
}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}

OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}
IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}

STATE_TO_IDX = {"open": 0, "closed": 1, "locked": 2}

# === Helper Functions ===

def symbolic_to_array(state, max_objects=10, max_walls=25):
    """
    Converts a symbolic state (a dict with keys "direction", "objects",
    "outer_walls", and "goal") into a 1D numpy array in the same format as
    produced by the PaddedObservationWrapper.
    
    The layout is:
      [direction,
       objects[0] (5 values), ..., objects[max_objects-1] (5 values),
       outer_walls[0] (2 values), ..., outer_walls[max_walls-1] (2 values),
       goal (2 values)]
    """
    obs_dim = 1 + 5 * max_objects + 2 * max_walls + 2
    obs_array = np.zeros((obs_dim,), dtype=np.float32)
    # Insert direction (index 0)
    obs_array[0] = float(state["direction"])
    # Insert objects (each with 5 values)
    for i, obj in enumerate(state["objects"][:max_objects]):
        start_idx = 1 + i * 5
        obs_array[start_idx:start_idx + 5] = np.array(obj, dtype=np.float32)
    # Insert outer walls (each with 2 values)
    wall_offset = 1 + max_objects * 5
    for i, wall in enumerate(state["outer_walls"][:max_walls]):
        start_idx = wall_offset + i * 2
        obs_array[start_idx:start_idx + 2] = np.array(wall, dtype=np.float32)
    # Insert goal (last 2 slots)
    goal_offset = wall_offset + max_walls * 2
    obs_array[goal_offset:goal_offset + 2] = np.array(state["goal"], dtype=np.float32)
    return obs_array


def state_to_key(state):
    """
    Converts a symbolic state (dict) into a hashable tuple.
    We include the agent's direction, goal, sorted objects, and sorted outer walls.
    """
    # Note: objects are assumed already sorted by our get_neighbors function.
    objects_key = tuple(tuple(obj) for obj in state["objects"])
    outer_walls_key = tuple(sorted(tuple(w) for w in state["outer_walls"]))
    return (state["direction"], tuple(state["goal"]), objects_key, outer_walls_key)


def get_grid_dimensions(state):
    """
    Computes grid width and height from the outer walls. Assumes that outer walls
    include all positions on the boundary.
    """
    if not state["outer_walls"]:
        return None, None
    max_x = max(x for (x, y) in state["outer_walls"])
    max_y = max(y for (x, y) in state["outer_walls"])
    return max_x + 1, max_y + 1


def get_agent_position(state):
    """
    Returns the (x, y) position of the agent, if present in state["objects"].
    """
    for obj in state["objects"]:
        if obj[0] == OBJECT_TO_IDX["agent"]:
            return (obj[3], obj[4])
    return None


def get_occupied_positions(state, exclude_idx=None):
    """
    Returns a set of positions (x, y) that are occupied by objects.
    If exclude_idx is provided, the object at that index in state["objects"]
    is not considered.
    """
    occupied = set()
    for i, obj in enumerate(state["objects"]):
        if i == exclude_idx:
            continue
        occupied.add((obj[3], obj[4]))
    return occupied


def get_neighbors(state, env_name, max_gen_objects=2, **kwargs):
    """
    Dispatch to the correct environment-specific neighbor function
    based on env_name.
    """
    if "fetch" in env_name.lower():
        return get_neighbors_fetch_old(state, max_gen_objects=max_gen_objects, **kwargs)
    elif "empty" in env_name.lower():
        return get_neighbors_empty(state, max_gen_objects=max_gen_objects, **kwargs)
    elif "doorkey" in env_name.lower():
        return get_neighbors_doorkey(state, max_gen_objects=max_gen_objects, **kwargs)
    else:
        raise ValueError(f"get_neighbors not implemented for env_name: {env_name}")


def get_neighbors_empty(state, max_gen_objects=10, **kwargs):
    """
    TODO: Implement neighbor-generation logic for 'MiniGrid-Empty' environments.
    Currently returns an empty list as a placeholder.
    """
    return []

def get_neighbors_doorkey(state, max_gen_objects=10, **kwargs):
    """
    TODO: Implement neighbor-generation logic for 'MiniGrid-DoorKey' environments.
    Currently returns an empty list as a placeholder.
    """
    return []


def get_neighbors_fetch_old(state, max_gen_objects, **kwargs):
    """
    Given a symbolic state (dict), returns a list of neighbor states that are
    exactly one atomic modification (L1 difference of 1) away.

    Atomic modifications include:
      - Changing agent direction by ±1 (if within bounds [0, 3])
      - Changing the goal's color (to any other allowed color) or type (to key/ball)
      - For each non-agent object:
            • Changing its color (to any other allowed color)
            • Changing its type (allowed types: key and ball)
            • Changing its state (0=open, 1=closed, 2=locked) **only if object is a door**
            • Changing its position by one step (only one coordinate, provided the new position is in bounds, not on an outer wall, and not colliding with another object)
      - Creating a new object (if len(objects) < max_gen_objects) with a full specification
        (type from {key, ball}, any allowed color, and a valid position)
    """

    neighbors = []
    # --- 1. Modify agent's direction ---
    current_direction = state["direction"]
    if current_direction > 0:
        new_state = copy.deepcopy(state)
        new_state["direction"] = current_direction - 1
        neighbors.append(new_state)
    if current_direction < 3:
        new_state = copy.deepcopy(state)
        new_state["direction"] = current_direction + 1
        neighbors.append(new_state)

    # --- 2. Modify goal ---
    current_goal = state["goal"]  # [obj_idx, color_idx]
    # Goal: modify color (allowed colors from COLOR_TO_IDX)
    for col_name, col_idx in COLOR_TO_IDX.items():
        if col_idx != current_goal[1]:
            new_state = copy.deepcopy(state)
            new_state["goal"][1] = col_idx
            neighbors.append(new_state)
    # Goal: modify type (allowed types: key and ball)
    allowed_goal_types = [OBJECT_TO_IDX["key"], OBJECT_TO_IDX["ball"]]
    for t in allowed_goal_types:
        if t != current_goal[0]:
            new_state = copy.deepcopy(state)
            new_state["goal"][0] = t
            neighbors.append(new_state)

    # --- 3. Modify existing objects ---
    grid_width, grid_height = get_grid_dimensions(state)
    if grid_width is None or grid_height is None:
        # Fallback default grid size (if outer_walls not available)
        grid_width, grid_height = 10, 10

    agent_pos = get_agent_position(state)

    for i, obj in enumerate(state["objects"]):
        obj_idx, color_idx, obj_state, x, y = obj
        # Do not modify the agent object
        if obj_idx == OBJECT_TO_IDX["agent"]:
            continue

        # a. Change object's color
        for col_name, col_idx in COLOR_TO_IDX.items():
            if col_idx != color_idx:
                new_state = copy.deepcopy(state)
                new_state["objects"][i][1] = col_idx
                new_state["objects"].sort(key=lambda o: (o[0], o[3], o[4]))
                neighbors.append(new_state)

        # b. Change object's type (allowed types: key and ball)
        allowed_types = [OBJECT_TO_IDX["key"], OBJECT_TO_IDX["ball"]]
        for t in allowed_types:
            if t != obj_idx:
                new_state = copy.deepcopy(state)
                new_state["objects"][i][0] = t
                new_state["objects"].sort(key=lambda o: (o[0], o[3], o[4]))
                neighbors.append(new_state)

        # c. Change object's state ONLY if it's a door
        #    (0=open, 1=closed, 2=locked)
        if obj_idx == OBJECT_TO_IDX["door"] and obj_state in [0, 1, 2]:
            if obj_state == 0:        # open
                possible_states = [1] # can go to closed
            elif obj_state == 1:      # closed
                possible_states = [0, 2] # open or locked
            elif obj_state == 2:      # locked
                possible_states = [1] # can go to closed

            for new_obj_state in possible_states:
                new_state = copy.deepcopy(state)
                new_state["objects"][i][2] = new_obj_state
                new_state["objects"].sort(key=lambda o: (o[0], o[3], o[4]))
                neighbors.append(new_state)

        # d. Change object's position (only one coordinate at a time)
        # Get positions occupied by all other objects
        occupied = get_occupied_positions(state, exclude_idx=i)
        # For x coordinate
        for dx in [-1, 1]:
            new_x = x + dx
            if 1 <= new_x <= grid_width - 2:
                if (new_x, y) not in occupied and (new_x, y) not in state["outer_walls"]:
                    if agent_pos is None or (new_x, y) != agent_pos:
                        new_state = copy.deepcopy(state)
                        new_state["objects"][i][3] = new_x
                        new_state["objects"].sort(key=lambda o: (o[0], o[3], o[4]))
                        neighbors.append(new_state)
        # For y coordinate
        for dy in [-1, 1]:
            new_y = y + dy
            if 1 <= new_y <= grid_height - 2:
                if (x, new_y) not in occupied and (x, new_y) not in state["outer_walls"]:
                    if agent_pos is None or (x, new_y) != agent_pos:
                        new_state = copy.deepcopy(state)
                        new_state["objects"][i][4] = new_y
                        new_state["objects"].sort(key=lambda o: (o[0], o[3], o[4]))
                        neighbors.append(new_state)

    # --- 4. Create a new object (if allowed) ---
    if len(state["objects"]) < max_gen_objects:
        # Only allow objects of type key or ball.
        allowed_types = [OBJECT_TO_IDX["key"], OBJECT_TO_IDX["ball"]]
        # Compute all valid positions in the interior (positions not on outer walls)
        grid_width, grid_height = get_grid_dimensions(state)
        grid_positions = [(x, y) for x in range(1, grid_width - 1) for y in range(1, grid_height - 1)]
        valid_positions = [p for p in grid_positions if p not in state["outer_walls"]]
        # Exclude positions already occupied by an object.
        occupied = get_occupied_positions(state)
        valid_positions = [p for p in valid_positions if p not in occupied]
        # For each allowed combination, add a new object in one step.
        for t in allowed_types:
            for col_name, col_idx in COLOR_TO_IDX.items():
                for pos in valid_positions:
                    new_state = copy.deepcopy(state)
                    default_state = 0  # default door state is not relevant here for key/ball
                    new_obj = [t, col_idx, default_state, pos[0], pos[1]]
                    new_state["objects"].append(new_obj)
                    new_state["objects"].sort(key=lambda o: (o[0], o[3], o[4]))
                    neighbors.append(new_state)

    return neighbors




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


# === BFS-RR Algorithm ===

def bfs_rr(
    initial_state,
    model,
    env_name,
    max_obs_objects=10,  # For symbolic_to_array / PaddedObservation
    max_walls=25,
    max_gen_objects=2,   # For neighbor generation only
    max_depth=10,
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
      - max_depth (int): maximum number of nodes to dequeue/expand in the BFS.

    Returns:
      - robustness_region (list): a list of symbolic states (each a dict) in the robustness region,
                                  in the order discovered.
      - stats (dict): dictionary containing statistics about the BFS, including:
          * "initial_action"
          * "region_size": number of states in the robustness region
          * "total_opened_nodes": total number of nodes expanded
          * "visited_count": total number of visited states
          * "elapsed_time": time taken (seconds)
    """
    import time
    from collections import deque
    
    start_time = time.time()
    total_opened_nodes = 0

    # 1. Get the initial action for the starting state
    initial_obs = symbolic_to_array(initial_state, max_obs_objects, max_walls)
    initial_action, _ = model.predict(initial_obs, deterministic=True)

    # 2. Initialize BFS structures
    region = {}              # Maps a state key -> state
    robustness_region = []   # States in the region (in order discovered)
    visited = set()          # Keys of states we've already visited
    queue = deque()

    # 3. Enqueue the initial state
    init_key = state_to_key(initial_state)
    visited.add(init_key)
    queue.append(initial_state)

    # 4. BFS
    while queue and total_opened_nodes < max_depth:
        total_opened_nodes += 1
        if len(visited) % 500 == 0:
            print(f"Debug: Visited {len(visited)} states; Queue size: {len(queue)}")

        state = queue.popleft()
        key = state_to_key(state)

        # Predict the action for this state
        obs = symbolic_to_array(state, max_obs_objects, max_walls)
        action, _ = model.predict(obs, deterministic=True)
        print(f"Debug: Predicted action for state {key}: {action}")
        print(f"Debug: Initial action: {initial_action}")
        print(f"Debug: Action match: {action == initial_action}")

        # If action matches, this state is in the region
        if action == initial_action:
            if key not in region:
                region[key] = state
                robustness_region.append(state)

            # Expand neighbors based on env-specific rules
            neighbors = get_neighbors(state, env_name=env_name, max_gen_objects=max_gen_objects)
            for neighbor in neighbors:
                nkey = state_to_key(neighbor)
                if nkey not in visited:
                    visited.add(nkey)
                    queue.append(neighbor)

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
    import os
    import yaml
    import datetime

    # Hardcode the model path (folder) to load the saved model.
    model_path = "data/experiments/models/MiniGrid-Fetch-5x5-N2-v0_PPO_model_20250305_031749"

    # Replace load_model with load_experiment.
    from utils import load_experiment
    model, config_data = load_experiment(model_path)
    env_config = config_data["env_config"]  # Extract the environment configuration

    # Set render_mode to human for rendering.
    env_config["render_mode"] = "human"

    print(f"Loaded model for environment: {env_config['env_name']}")
    print(f"Configuration: {env_config}")

    # --- Create the symbolic MiniGrid environment ---
    from environment_utils import create_minigrid_env
    env = create_minigrid_env(env_config)
    symbolic_env = get_symbolic_env(env)

    # Use a fixed seed value
    seed_value = 42

    # Reset the environment to get the initial symbolic state.
    initial_symbolic_state, info = symbolic_env.reset(seed=seed_value, options={})
    env.render()

    # Reset the full environment.
    initial_state, info = env.reset(seed=seed_value, options={})
    env.render()

    action, _ = model.predict(initial_state, deterministic=True)
    print(f"Initial action: {action}")
    print(f"Initial symbolic state: {initial_symbolic_state}")
    print(f"Initial state: {initial_state}")

    # --- Calculate the Robustness Region ---
    robustness_region, stats = bfs_rr(initial_symbolic_state, model,
                                        env_name=env_config["env_name"],
                                        max_obs_objects=env_config["max_objects"],
                                        max_walls=env_config["max_walls"],
                                        max_gen_objects=max_gen_objects,
                                        max_depth=10)

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

    env.close()

