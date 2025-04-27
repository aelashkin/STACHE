#!/usr/bin/env python3
"""Taxi‑v3 SB3‑policy visualizer

This script
1. loads a deterministic Stable‑Baselines3 model trained on *Taxi‑v3* with a
   one‑hot observation wrapper;
2. queries the policy for **every one of the 500 discrete states** in the
   environment;
3. stores the resulting state‑→action mapping as a **YAML** file in the
   experiment folder structure requested by the user; and
4. builds colour‑blind‑friendly **PNG** visualisations that show, for each
   destination, the action the policy would take from every taxi position in
   (a) the three possible "passenger waiting" configurations and (b) the
   corresponding "passenger in taxi" configuration.

Usage (from the command line) ───────────────────────────────────────────────

    python taxi_policy_visualization.py \
        --model-path data/models/taxi_dqn.zip \
        --model-name taxi_dqn \
        [--timestamp 20250423_153045]  # optional; generated if omitted

The outputs are written to

    data/experiments/rr/<model_name>/_time_<timestamp>/

Dependencies
------------
* gymnasium >= 1.0.0
* stable‑baselines3 >= 2.0.0
* pyyaml
* matplotlib

The script is fully self‑contained and PEP 8 compliant.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap
from stable_baselines3 import DQN

# ──────────────────────────────────────────────────────────────────────────────
# Constants & colour palette (colour‑blind friendly)
# ──────────────────────────────────────────────────────────────────────────────

ACTION_NAMES = {
    0: "South",
    1: "North",
    2: "East",
    3: "West",
    4: "Pickup",
    5: "Dropoff",
}

CB_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish‑green
    "#CC79A7",  # reddish‑purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
]

_COLORMAP = ListedColormap(CB_PALETTE, name="taxi_actions")

# Map locations as (row, col)
PICKUP_LOCS = {
    0: (0, 0),  # R
    1: (0, 4),  # G
    2: (4, 0),  # Y
    3: (4, 3),  # B
}   

# Character representation for labelling
LOC_CHARS = {0: "R", 1: "G", 2: "Y", 3: "B"}

# ──────────────────────────────────────────────────────────────────────────────
# One‑hot observation wrapper (unchanged from training)
# ──────────────────────────────────────────────────────────────────────────────

class OneHotObs(gym.ObservationWrapper):
    """Convert Discrete(N) observation to one‑hot float32 vector length N."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(
            env.observation_space, gym.spaces.Discrete
        ), "OneHotObs only supports Discrete spaces"
        n = env.observation_space.n
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(n,), dtype=np.float32
        )

    def observation(self, obs: int) -> np.ndarray:  # type: ignore[override]
        vec = np.zeros(self.observation_space.shape, dtype=np.float32)
        vec[obs] = 1.0
        return vec

# ──────────────────────────────────────────────────────────────────────────────
# Core functionality
# ──────────────────────────────────────────────────────────────────────────────

def collect_state_actions(model: DQN, env: gym.Env, base_env: gym.Env) -> Dict[int, int]:
    """Query *model* once for **every** discrete state in *base_env*.

    Parameters
    ----------
    model : DQN
        The trained SB3 model.
    env : gym.Env
        The *wrapped* environment (e.g., OneHotObs) compatible with the model.
    base_env : gym.Env
        The *unwrapped* base environment (e.g., Taxi-v3) to get state count.

    Returns
    -------
    mapping : dict[int, int]
        Dictionary ``{state_id: chosen_action}``.
    """
    mapping: Dict[int, int] = {}
    num_states = base_env.observation_space.n # Use base_env here
    # Ensure the wrapped env's space matches the base env's discrete count
    assert isinstance(env.observation_space, gym.spaces.Box) and \
           env.observation_space.shape == (num_states,)

    for state in range(num_states):
        # Create the one-hot observation expected by the model
        obs = np.zeros((1,) + env.observation_space.shape, dtype=np.float32)
        obs[0, state] = 1.0
        action, _ = model.predict(obs, deterministic=True)
        mapping[state] = int(action[0])
    return mapping


def save_mapping_yaml(mapping: Dict[int, int], filepath: Path) -> None:
    """Save the state‑action mapping as YAML."""
    with filepath.open("w", encoding="utf‑8") as f:
        yaml.safe_dump(mapping, f, sort_keys=True)


def build_action_grid(
    taxi_env: gym.Env, mapping: Dict[int, int], passenger_loc: int, dest_idx: int
) -> np.ndarray:
    """Create a 5×5 grid where each cell holds the policy action (0‑5).

    Parameters
    ----------
    taxi_env : gym.Env
        *Unwrapped* Taxi‑v3 environment (for encode())
    mapping : dict[int, int]
        State‑action lookup produced by :pyfunc:`collect_state_actions`.
    passenger_loc : int
        Passenger location index (0‑3 for waiting, 4 for in‑taxi).
    dest_idx : int
        Destination index (0‑3).

    Returns
    -------
    grid : ndarray shape (5, 5)
        ``grid[row, col]`` → action id (int) for that taxi position.
    """
    grid = np.full((5, 5), fill_value=-1, dtype=int)
    for row in range(5):
        for col in range(5):
            state = taxi_env.encode(row, col, passenger_loc, dest_idx)  # type: ignore[attr-defined]
            grid[row, col] = mapping[state]
    return grid


def _annotate_grid(ax, grid: np.ndarray, passenger: Tuple[int, int] | None, dest: Tuple[int, int], show_walls: bool = True) -> None:
    """Overlay P/G labels and walls on an imshow‑based grid."""
    ax.set_xticks([])
    ax.set_yticks([])
    for row in range(5):
        for col in range(5):
            label = ""
            if passenger and (row, col) == passenger:
                label = "P"
            if (row, col) == dest:
                label = "D" if not label else "PD"
            if label:
                ax.text(
                    col,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize="medium",
                    color="black",
                    weight="bold",
                )
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)

    # Draw walls if requested
    if show_walls:
        wall_kwargs = {'color': 'black', 'linewidth': 2.5}
        # Vertical walls
        ax.plot([0.5, 0.5], [2.5, 4.5], **wall_kwargs) # Between col 0 & 1, rows 3-4
        ax.plot([1.5, 1.5], [-0.5, 1.5], **wall_kwargs) # Between col 1 & 2, rows 0-1
        ax.plot([2.5, 2.5], [-0.5, 1.5], **wall_kwargs) # Between col 2 & 3, rows 0-1
        ax.plot([3.5, 3.5], [2.5, 4.5], **wall_kwargs) # Between col 3 & 4, rows 3-4


def plot_dest_maps(
    taxi_env: gym.Env,
    mapping: Dict[int, int],
    dest_idx: int,
    output_path: Path,
    show_walls: bool = True,
) -> None:
    """Render the 4‑panel visualisation for a given destination."""
    waiting_locs = [i for i in range(4) if i != dest_idx]
    passenger_configs = waiting_locs + [4]  # + in‑taxi

    # Increase figure height slightly to accommodate legend below
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for ax, pass_loc in zip(axes, passenger_configs):
        grid = build_action_grid(taxi_env, mapping, pass_loc, dest_idx)
        im = ax.imshow(grid, cmap=_COLORMAP, vmin=0, vmax=5)

        pickup_cell = PICKUP_LOCS[pass_loc] if pass_loc < 4 else None
        dest_cell = PICKUP_LOCS[dest_idx]
        _annotate_grid(ax, grid, pickup_cell, dest_cell, show_walls=show_walls)

        subtitle = (
            f"Passenger at {LOC_CHARS[pass_loc]}" if pass_loc < 4 else "Passenger in taxi"
        )
        ax.set_title(subtitle, fontsize="small")

    # Single legend for the whole figure, placed below axes
    legend_elems = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=CB_PALETTE[a], label=ACTION_NAMES[a])
        for a in range(6)
    ]
    fig.legend(
        handles=legend_elems,
        loc='upper center',  # Anchor point of the legend box
        bbox_to_anchor=(0.5, 0.05),  # Position: horizontal center, lower down
        ncol=6,
        title="Action taken by policy",
        fontsize="small",
    )

    # Adjust layout to prevent overlap, leaving space at the bottom
    fig.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust bottom margin if needed

    fig.suptitle(f"Destination = {LOC_CHARS[dest_idx]}", y=1.05)  # Adjust title position if needed
    plt.savefig(output_path, dpi=150, bbox_inches='tight')  # Use bbox_inches='tight' for better saving
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# High‑level orchestration
# ──────────────────────────────────────────────────────────────────────────────

def run_visualisation(model_path: Path, model_name: str, timestamp: str | None = None, show_walls: bool = True) -> None:
    """Main entry point for policy-map visualisation."""
    timestamp = timestamp or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Construct output directory for policy maps
    rr_dir = Path.cwd() / "data" / "experiments" / "rr" / f"{model_name}_policy_map" / f"time_{timestamp}"
    rr_dir.mkdir(parents=True, exist_ok=True)

    # Environment compatible with the trained policy (one‑hot)
    base_env = gym.make("Taxi-v3")
    env = OneHotObs(base_env)  # type: ignore[arg-type]

    # Load SB3 model and set the inference environment directly
    model = DQN.load(model_path, env=env, print_system_info=False)

    # 1. Collect actions for all states
    # Pass both the wrapped env (for prediction) and base_env (for state iteration)
    mapping = collect_state_actions(model, env, base_env)

    # 2. Save YAML
    yaml_path = rr_dir / "state_action_mapping.yaml"
    save_mapping_yaml(mapping, yaml_path)
    # Now yaml_path is absolute because rr_dir is absolute
    print(f"✔ Saved mapping → {yaml_path.relative_to(Path.cwd())}")

    # 3. Build visualisations
    unwrapped_env = base_env.unwrapped  # for encode/decode
    for dest in range(4):
        img_path = rr_dir / f"policy_map_dest_{LOC_CHARS[dest]}.png"
        plot_dest_maps(unwrapped_env, mapping, dest, img_path, show_walls=show_walls)
        # Now img_path is absolute because rr_dir is absolute
        print(f"✔ Saved visualisation → {img_path.relative_to(Path.cwd())}")

    # --- Combined 4x4 policy map ---
    dest_order = [3, 1, 0, 2]  # B, G, R, Y
    def passenger_list(d):
        return [p for p in range(5) if p != d][:3] + [4]
    fig4, axes4 = plt.subplots(4, 4, figsize=(16, 16))
    for row_idx, d in enumerate(dest_order):
        for col_idx, p in enumerate(passenger_list(d)):
            ax = axes4[row_idx, col_idx]
            grid = build_action_grid(unwrapped_env, mapping, p, d)
            ax.imshow(grid, cmap=_COLORMAP, vmin=0, vmax=5)
            pickup = PICKUP_LOCS[p] if p < 4 else None
            dest_cell = PICKUP_LOCS[d]
            _annotate_grid(ax, grid, pickup, dest_cell, show_walls=show_walls)
    # Shared legend
    legend_elems = [plt.Line2D([0], [0], marker="s", linestyle="", color=CB_PALETTE[a], label=ACTION_NAMES[a]) for a in range(len(ACTION_NAMES))]
    fig4.legend(handles=legend_elems, loc="lower center", ncol=6, title="Action", fontsize="small")
    fig4.tight_layout(rect=[0, 0.05, 1, 1])
    fig4.suptitle("Combined Policy Maps B-G-R-Y", y=1.02)
    out4_path = rr_dir / "policy_map_4x4.png"
    fig4.savefig(out4_path, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"✔ Saved combined 4x4 visualisation → {out4_path.relative_to(Path.cwd())}")

    print("All done! ✨")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taxi‑v3 policy visualiser")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/experiments/models/Taxi-v3_DQN_model_20250423_173106/model.zip"),
        help="Path to the SB3 .zip model file (produced by model.save()). Default: data/experiments/models/Taxi-v3_DQN_model_20250423_173106/model.zip",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Taxi-v3_DQN",
        help="Model identifier – used to build the rr experiment directory. Default: Taxi-v3_DQN",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Use a fixed timestamp instead of the current datetime.",
    )
    parser.add_argument(
        "--hide-walls",
        action="store_false",
        dest="show_walls", # Store result in 'show_walls'
        help="Do not draw the environment walls on the plots.",
    )
    parser.set_defaults(show_walls=True) # Default is to show walls
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    run_visualisation(args.model_path, args.model_name, args.timestamp, show_walls=args.show_walls)


if __name__ == "__main__":  # pragma: no cover
    main()
