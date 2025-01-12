import gymnasium as gym
from gymnasium.core import ObservationWrapper
from gymnasium.spaces import Dict, Discrete, Box
from gymnasium import spaces
import numpy as np

# Reuse your mappings
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

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
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}


def extract_objects_from_state(grid):
    """
    Convert fully observable grid representation into a sorted list of objects
    with their state and positions (x, y), EXCLUDING outer walls.

    Format: [OBJ_IDX, COLOR_IDX, OBJ_STATE, x, y]
    """
    height, width, _ = grid.shape
    object_list = []

    # Traverse the interior of the grid (ignoring outer walls)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            obj_idx, color_idx, obj_state = grid[y, x]
            # Skip 'empty' or 'unseen'
            if obj_idx in [OBJECT_TO_IDX["empty"], OBJECT_TO_IDX["unseen"]]:
                continue

            object_list.append([obj_idx, color_idx, obj_state, x, y])

    # Sort by obj_idx (descending), then by (y, x)
    object_list.sort(key=lambda obj: (-int(obj[0]), obj[4], obj[3]))
    return object_list


def extract_outer_walls(grid):
    """
    Extract the positions (x, y) of the outer walls.
    """
    height, width, _ = grid.shape
    wall_positions = []

    # Top and bottom rows
    for x in range(width):
        top_idx, top_color, top_state = grid[0, x]
        if top_idx == OBJECT_TO_IDX["wall"]:
            wall_positions.append((x, 0))

        bottom_idx, bottom_color, bottom_state = grid[height - 1, x]
        if bottom_idx == OBJECT_TO_IDX["wall"]:
            wall_positions.append((x, height - 1))

    # Left and right columns
    for y in range(height):
        left_idx, left_color, left_state = grid[y, 0]
        if left_idx == OBJECT_TO_IDX["wall"]:
            wall_positions.append((0, y))

        right_idx, right_color, right_state = grid[y, width - 1]
        if right_idx == OBJECT_TO_IDX["wall"]:
            wall_positions.append((width - 1, y))

    return wall_positions


class FactorizedSymbolicWrapper(ObservationWrapper):
    """
    Wraps a fully observable MiniGrid environment (like after FullyObsWrapper)
    to produce a dict observation with:
      {
          "direction": int in [0..3],
          "objects": list of [obj_idx, color_idx, state, x, y],
          "outer_walls": list of (x, y),
          "goal": [obj_idx, color_idx]  # (if a goal is parsed)
      }
    """

    def __init__(self, env):
        super().__init__(env)

        # Unwrap the environment to the base
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        self.base_env = unwrapped_env  # Store the unwrapped base environment

        # Keep the original action space
        self.action_space = env.action_space

        # Define the observation space
        self.observation_space = Dict({
            "direction": Discrete(4),
            "objects": gym.spaces.Sequence(gym.spaces.Box(
                low=0, high=255, shape=(5,), dtype=np.int32
            )),
            "outer_walls": gym.spaces.Sequence(gym.spaces.Box(
                low=0, high=255, shape=(2,), dtype=np.int32
            )),
            "goal": gym.spaces.Box(
                low=0, high=255, shape=(2,), dtype=np.int32
            ),
        })

    def parse_goal(self, mission_str):
        """
        Parse the mission to determine the goal object and color.
        """
        # Default to 'unseen' object and 'grey' color if no match is found
        goal_obj_idx = OBJECT_TO_IDX["unseen"]
        goal_color_idx = COLOR_TO_IDX["grey"]

        # Only parse if environment is named something like "FetchEnv"
        if self.base_env.__class__.__name__ == "FetchEnv":
            # Example mission: "go fetch a purple key"
            tokens = mission_str.lower().split()

            # If we assume the mission always ends with "<color> <type>"
            if len(tokens) >= 2:
                color_token = tokens[-2].strip(",.!?")
                type_token = tokens[-1].strip(",.!?")

                # Look up the color index
                if color_token in COLOR_TO_IDX:
                    goal_color_idx = COLOR_TO_IDX[color_token]

                # Look up the object index
                if type_token in OBJECT_TO_IDX:
                    goal_obj_idx = OBJECT_TO_IDX[type_token]

        return [goal_obj_idx, goal_color_idx]

    def observation(self, obs):
        """
        Expects obs to be something like:
          {
            'direction': 0..3,
            'image': (height, width, 3),
            'mission': str
          }
        """
        direction = obs["direction"]
        grid_img = obs["image"]
        mission_str = obs.get("mission", "")

        # Create the object list (excluding outer walls)
        objects = extract_objects_from_state(grid_img)

        # Create a list of outer wall positions
        outer_walls = extract_outer_walls(grid_img)

        # Determine the goal (e.g., for fetch envs)
        goal = self.parse_goal(mission_str)  # Use the method here

        return {
            "direction": direction,
            "objects": objects,
            "outer_walls": outer_walls,
            "goal": goal,
        }





class PaddedObservationWrapper(gym.ObservationWrapper):
    """
    Converts the dictionary observation from FactorizedSymbolicWrapper
    into a fixed-size 1D numpy array for consumption by SB3 PPO.

    The resulting observation is structured as follows (concatenated into 1D):

      [direction,
       objects[0], objects[1], ..., objects[MAX_OBJECTS-1],
       outer_walls[0], ..., outer_walls[MAX_WALLS-1],
       goal_obj_idx, goal_color_idx]

    If there are fewer objects than MAX_OBJECTS, we pad the rest with 0s.
    If there are more objects, we truncate.

    Same for outer walls.
    """
    def __init__(self, env, max_objects=10, max_walls=25):
        super().__init__(env)

        self.max_objects = max_objects
        self.max_walls = max_walls

        # Each object has 5 values: obj_idx, color_idx, state, x, y
        # We'll store them as float32 for convenience in neural nets.
        object_dim = 5 * self.max_objects

        # Each wall has 2 values (x, y)
        wall_dim = 2 * self.max_walls

        # We'll store direction as 1 dimension, and the goal as 2 dimensions.
        # So total = direction (1) + objects (object_dim) + walls (wall_dim) + goal (2).
        self.obs_dim = 1 + object_dim + wall_dim + 2

        # We define an arbitrary bounding box for all values,
        # e.g. 0..255 or 0..100. We just need something that covers possible
        # indices and coordinates. You can adjust if needed for bigger grids.
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

    def observation(self, obs):
        """
        Obs is a dict:
          {
            "direction": int in [0..3],
            "objects": list of [obj_idx, color_idx, obj_state, x, y],
            "outer_walls": list of (x, y),
            "goal": [obj_idx, color_idx]
          }
        We convert it into a float32 array of size self.obs_dim.
        """
        direction = obs["direction"]
        objects = obs["objects"]
        outer_walls = obs["outer_walls"]
        goal = obs["goal"]

        # Prepare the final array
        obs_array = np.zeros((self.obs_dim,), dtype=np.float32)

        # 1) Insert direction (index 0)
        obs_array[0] = float(direction)

        # 2) Insert objects (up to max_objects)
        # Each object is 5 values
        # objects[i] = [obj_idx, color_idx, state, x, y]
        for i, obj_info in enumerate(objects[: self.max_objects]):
            start_idx = 1 + i * 5
            obs_array[start_idx : start_idx + 5] = obj_info

        # 3) Insert outer_walls (up to max_walls)
        # each wall is (x, y)
        wall_offset = 1 + self.max_objects * 5
        for i, wall_xy in enumerate(outer_walls[: self.max_walls]):
            start_idx = wall_offset + i * 2
            obs_array[start_idx : start_idx + 2] = wall_xy

        # 4) Insert goal (last 2 slots: obj_idx, color_idx)
        goal_offset = wall_offset + self.max_walls * 2
        obs_array[goal_offset : goal_offset + 2] = goal

        return obs_array