#!/usr/bin/env python
"""
This file defines MiniGridEnvExt, an extension of the MiniGridEnv class that supports
initializing the environment with a provided initial state. An initial state is expected
to be a dictionary with the following keys:
  - "grid": a serialized grid (a numpy array of shape (width, height, 3) as produced by Grid.encode())
  - "agent_pos": a tuple of ints indicating the agent's position
  - "agent_dir": an int (0-3) indicating the agent's direction
  - "mission": a string specifying the mission
  - "carrying": (optional) a serialized object or None

If no initial state is provided, the environment behaves as usual.
A main() function is provided at the end to demonstrate a running example.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType
import time
import pygame

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import TILE_PIXELS
from minigrid.core.grid import Grid


class MiniGridEnvExt(MiniGridEnv):
    def __init__(
        self,
        mission_space: MissionSpace,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        initial_state: dict[str, Any] | None = None,  # NEW parameter for initial state
    ):
        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agent_view_size=agent_view_size,
            render_mode=render_mode,
            screen_size=screen_size,
            highlight=highlight,
            tile_size=tile_size,
            agent_pov=agent_pov,
        )
        # Store the provided initial state (if any)
        self.initial_state = initial_state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        # Reset episode-specific variables
        self.step_count = 0
        self.carrying = None

        # Use the provided initial state if available; otherwise, generate a new grid.
        if self.initial_state is not None:
            self.set_state(self.initial_state)
        else:
            self.agent_pos = (-1, -1)
            self.agent_dir = -1
            self._gen_grid(self.width, self.height)

        # Ensure that the agent's position and direction have been set
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        if self.render_mode == "human":
            self.render()

        # Return the initial observation
        obs = self.gen_obs()
        return obs, {}

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Set the environment's state from the provided serialized state dictionary.

        Expected structure for 'state':
          {
            "grid": <serialized grid (as produced by Grid.encode())>,
            "agent_pos": <tuple of ints>,
            "agent_dir": <int>,
            "mission": <str>,
            "carrying": <serialized object or None>  # optional
          }
        """
        # Reconstruct the grid from its serialized representation.
        self.grid, _ = Grid.decode(state["grid"])
        self.agent_pos = tuple(state["agent_pos"])
        self.agent_dir = state["agent_dir"]
        self.mission = state["mission"]
        self.carrying = state.get("carrying", None)

def main():
    """
    Running example for MiniGridEnvExt.
    
    This example manually creates a serialized grid for a 5x5 environment with the following layout:
      - Border cells are walls.
      - Interior cells are empty except:
          * A blue key is placed at position (2,2)
          * A red ball is placed at position (1,2)
      - The agent starts at position (3,3) facing right (direction 0).
      - The mission is "get a blue key".
    """
    # Standard mappings from minigrid
    COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
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
    STATE_TO_IDX = {"open": 0, "closed": 1, "locked": 2}

    # Encoded representations for objects.
    wall = np.array([OBJECT_TO_IDX["wall"], COLOR_TO_IDX["grey"], STATE_TO_IDX["open"]], dtype=np.uint8)
    empty = np.array([OBJECT_TO_IDX["empty"], 0, 0], dtype=np.uint8)
    blue_key = np.array([OBJECT_TO_IDX["key"], COLOR_TO_IDX["blue"], 0], dtype=np.uint8)
    red_ball = np.array([OBJECT_TO_IDX["ball"], COLOR_TO_IDX["red"], 0], dtype=np.uint8)

    # Create a 5x5 grid encoding (shape: (width, height, 3)).
    grid_width, grid_height = 5, 5
    serialized_grid = np.empty((grid_width, grid_height, 3), dtype=np.uint8)

    # Set border cells to walls and interior cells to empty.
    for i in range(grid_width):
        for j in range(grid_height):
            if i == 0 or i == grid_width - 1 or j == 0 or j == grid_height - 1:
                serialized_grid[i, j, :] = wall
            else:
                serialized_grid[i, j, :] = empty

    # Place a blue key at position (2,2) and a red ball at position (1,2)
    serialized_grid[2, 2, :] = blue_key
    serialized_grid[1, 2, :] = red_ball

    # Define the initial state.
    initial_state = {
        "grid": serialized_grid,
        "agent_pos": (3, 3),
        "agent_dir": 0,  # Facing right
        "mission": "get a blue key",
        "carrying": None,
    }

    # Define a mission function matching the FetchEnv pattern.
    def mission_func(syntax: str, color: str, obj_type: str):
        return f"{syntax} {color} {obj_type}"

    # Define the mission space with three ordered placeholders.
    mission_space = MissionSpace(
        mission_func=mission_func,
        ordered_placeholders=[
            ["get a", "go get a", "fetch a", "go fetch a", "you must fetch a"],
            ["blue"],
            ["key"]
        ]
    )

    # Create the environment using MiniGridEnvExt with the provided initial state.
    env = MiniGridEnvExt(
        mission_space=mission_space,
        width=5,
        height=5,
        render_mode="human",  # Use human mode to display the window.
        initial_state=initial_state,
    )

    # Reset the environment to load the initial state.
    obs, _ = env.reset()
    print("Initial observation:")
    print(obs)

    print("Rendering environment. Close the pygame window to exit.")
    running = True
    while running:
        # Process pygame events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        env.render()
        time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    main()
