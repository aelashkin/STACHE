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
import re
import math
import time
from collections import deque
from PIL import Image

import yaml
import gymnasium as gym
import numpy as np

from utils.experiment_io import load_experiment
from minigrid_ext.environment_utils import create_minigrid_env
from minigrid_ext.set_state_extension import set_standard_state_minigrid, factorized_symbolic_to_fullobs

# variables for running main without console input

# Hardcode the model path (folder) to load the saved model.
model_path = "data/experiments/models/MiniGrid-Empty-Random-6x6-v0_PPO_model_20250304_220518"

# model_path = "data/experiments/models/MiniGrid-Fetch-5x5-N2-v0_PPO_model_20250305_031749"
max_gen_objects = 2
max_nodes_expanded = None
seed_value = 77

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

# Hardcoded action values for "empty" and "fetch" environments
ACTION_MAPPING_EMPTY = [
    (0, "left", "Turn left"),
    (1, "right", "Turn right"),
    (2, "forward", "Move forward"),
    (3, "pickup", "Unused"),
    (4, "drop", "Unused"),
    (5, "toggle", "Unused"),
    (6, "done", "Unused"),
]

ACTION_MAPPING_FETCH = [
    (0, "left", "Turn left"),
    (1, "right", "Turn right"),
    (2, "forward", "Move forward"),
    (3, "pickup", "Pick up an object"),
    (4, "drop", "Unused"),
    (5, "toggle", "Unused"),
    (6, "done", "Unused"),
]

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
    Dispatch to the correct environment-specific neighbor function based on env_name.
    """
    if "fetch" in env_name.lower():
        return get_neighbors_fetch_old(state, max_gen_objects=max_gen_objects, **kwargs)
    elif "empty" in env_name.lower():
        dims = get_env_dimensions_from_name(env_name)
        return get_neighbors_empty(state, env_dimensions=dims, **kwargs)
    elif "doorkey" in env_name.lower():
        return get_neighbors_doorkey(state, max_gen_objects=max_gen_objects, **kwargs)
    else:
        raise ValueError(f"get_neighbors not implemented for env_name: {env_name}")


def get_neighbors_empty(state, env_dimensions=None, **kwargs):
    """
    Given a symbolic state (dict) for an EmptyEnv, returns a list of neighbor states
    that are exactly one atomic modification away. The only allowed modifications are:
      - Moving the agent's position by 1 step (up/down/left/right) within the valid interior.
      - Changing the agent's direction by ±1, if still in [0,3].
    
    Notes:
    - The 'goal' remains fixed in the bottom-right corner.
    - No new objects are generated.
    - If env_dimensions is provided as a (width, height) tuple, it is used; otherwise,
      grid dimensions are computed from the outer walls or defaulted to 8x8.
    """

    neighbors = []

    # --- 1. Modify agent's direction ---
    current_direction = state["direction"]
    # Direction wraps around [0, 3] interval
    new_directions = ((current_direction - 1) % 4, (current_direction + 1) % 4)
    for d in new_directions:
        new_state = copy.deepcopy(state)
        new_state["direction"] = d
        neighbors.append(new_state)

    # --- 2. Move the agent’s position by ±1 in x or y ---
    # Find the agent object in 'objects'
    agent_index = None
    for i, obj in enumerate(state["objects"]):
        if obj[0] == OBJECT_TO_IDX["agent"]:
            agent_index = i
            break

    # If no agent object found, no positional moves can be made
    if agent_index is None:
        raise ValueError("Agent object not found in state['objects']")

    # Use env_dimensions if provided; otherwise, compute from state or default to 8x8
    if env_dimensions is None:
        raise ValueError("env_dimensions must be provided for EmptyEnv.")
    else:
        grid_width, grid_height = env_dimensions

    x, y = state["objects"][agent_index][3], state["objects"][agent_index][4]
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in deltas:
        new_x = x + dx
        new_y = y + dy
        # Check if inside the valid interior: not on or outside the outer walls
        if 1 <= new_x <= grid_width - 2 and 1 <= new_y <= grid_height - 2:
            if (new_x, new_y) not in state["outer_walls"]:
                new_state = copy.deepcopy(state)
                new_state["objects"][agent_index][3] = new_x
                new_state["objects"][agent_index][4] = new_y
                # Sort objects for consistent state comparison
                new_state["objects"].sort(key=lambda o: (o[0], o[3], o[4]))
                neighbors.append(new_state)

    return neighbors


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
    # Direction wraps around [0, 3] interval
    new_directions = ((current_direction - 1) % 4, (current_direction + 1) % 4)
    for d in new_directions:
        new_state = copy.deepcopy(state)
        new_state["direction"] = d
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


def get_env_dimensions_from_name(env_name):
    """
    Extracts grid dimensions (width, height) from the env_name.
    Expects a pattern like 'empty-5x5'. Raises ValueError if not found.
    """
    match = re.search(r'(\d+)x(\d+)', env_name)
    if not match:
        raise ValueError("Invalid env name format. Expecting dimensions like 'empty-5x5'.")
    width = int(match.group(1))
    height = int(match.group(2))
    return (width, height)


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
