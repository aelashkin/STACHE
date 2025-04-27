import re
import numpy as np

from stache.envs.minigrid_ext.constants import (
    OBJECT_TO_IDX,
)


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
