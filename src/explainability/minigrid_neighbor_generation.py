import copy

from minigrid_ext.constants import (
    COLOR_TO_IDX,
    OBJECT_TO_IDX,
)
from minigrid_ext.state_utils import (
    get_grid_dimensions,
    get_agent_position,
    get_occupied_positions,
    get_env_dimensions_from_name,
)


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
