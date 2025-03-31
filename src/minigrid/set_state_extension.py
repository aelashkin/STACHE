import numpy as np
import gymnasium as gym
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.grid import Grid


def set_standard_state_minigrid(env, full_obs):
    """
    Set the state of a MiniGrid environment (env) from a fully observable
    observation (full_obs). The observation is assumed to come from a
    FullyObsWrapper or similar, with keys:
      - 'image':  np.array of shape (width, height, 3)
      - 'direction': int in [0..3]
      - 'mission': string

    The agent's step_count is left unchanged.

    After calling this function, env.grid, env.agent_pos, env.agent_dir,
    and env.mission will be updated accordingly.

    NOTE:
    - This assumes only a single agent in the grid.
    - If an 'agent' cell is found in multiple places, we use the first one
      encountered to set the agent's position.
    - We rely on 'direction' from the dictionary for the agent's direction
      (rather than the 'agent' cell's state).
    """
    # Unwrap the environment
    env = env.unwrapped

    # 1) Decode the grid from the 'image' array
    img = full_obs["image"]  # shape: (width, height, 3)
    new_grid, _ = Grid.decode(img)
    env.grid = new_grid

    # 2) Find the agent cell
    #    Note that 'img' has shape (width, height, 3), where the first dimension
    #    corresponds to x, and the second dimension corresponds to y.
    width, height = img.shape[0], img.shape[1]
    agent_pos = None

    for y in range(height):
        for x in range(width):
            cell_type, _, _ = img[x, y]
            if cell_type == OBJECT_TO_IDX["agent"]:
                agent_pos = (y, x)
                break
        if agent_pos is not None:
            break

    if agent_pos is None:
        raise ValueError("No agent cell found in the supplied observation.")

    # 3) Set the agent position and direction
    env.agent_pos = agent_pos
    env.agent_dir = full_obs["direction"]

    # 4) Update the mission string only if env.mission is currently empty
    if not env.mission:  # or env.mission.strip() == ''
        env.mission = full_obs.get("mission", "")





class SetStateWrapper(gym.Wrapper):
    """
    A wrapper that exposes a set_state(obs) method to inject
    a new environment state from a fully observable observation.
    """

    def __init__(self, env):
        super().__init__(env)

    def set_state(self, obs):
        """
        Set the underlying env's state according to the
        set_state_minigrid helper.
        """
        set_standard_state_minigrid(self.env, obs)




def factorized_symbolic_to_fullobs(fact_obs, width, height, mission=""):
    """
    Convert a FactorizedSymbolicWrapper-style observation back into the standard
    fully-observable observation dictionary that set_state expects, i.e.:

        {
            'image': np.array(shape=(width, height, 3), dtype=np.uint8),
            'direction': int in [0..3],
            'mission': <string>   # or an empty string if not provided
        }

    We assume:
      - fact_obs has keys: 'direction', 'objects', 'outer_walls', 'goal'
      - 'objects' is a list of [obj_idx, color_idx, state, x, y]
      - 'outer_walls' is a list of (x, y), each cell is presumably a Wall
      - We have the intended grid width, height externally, e.g. from env.width/height
      - The mission is kept the same or set to "" if not available
    """
    direction = fact_obs["direction"]
    # Create an empty 'image' array, defaulting to 'empty' object.
    # The image is created with shape (width, height, 3) so that the first index
    # corresponds to the x-coordinate and the second to the y-coordinate.
    image = np.zeros((width, height, 3), dtype=np.uint8)
    image[:, :, 0] = 1  # OBJECT_TO_IDX["empty"] == 1 by default

    # Mark outer walls as object=2 ('wall'), color=5 ('grey'), state=0.
    for (wx, wy) in fact_obs["outer_walls"]:
        if 0 <= wx < width and 0 <= wy < height:
            # Index using (wx, wy) since first dimension is x and second is y.
            image[wx, wy, 0] = 2  # OBJECT_TO_IDX["wall"]
            image[wx, wy, 1] = 5  # COLOR_TO_IDX["grey"]
            image[wx, wy, 2] = 0  # state=0

    # Place each object from the 'objects' list.
    for obj_idx, color_idx, state, ox, oy in fact_obs["objects"]:
        if 0 <= ox < width and 0 <= oy < height:
            image[ox, oy, 0] = obj_idx
            image[ox, oy, 1] = color_idx
            image[ox, oy, 2] = state

    # Use mission from fact_obs if available.
    mission = fact_obs.get("mission", mission)

    return {
        "image": image,
        "direction": direction,
        "mission": mission,
    }
