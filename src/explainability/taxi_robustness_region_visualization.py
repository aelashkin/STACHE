#!/usr/bin/env python3
"""
CLI for computing and visualizing the robustness region (RR) of a Taxi-v3 policy.
"""
import argparse
import datetime as _dt
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import DQN

from explainability.robust_taxi import compute_rr_taxi, translate_tuple_to_onehot
from explainability.taxi_robustness_region import OneHotObs, _annotate_grid, _COLORMAP, CB_PALETTE, ACTION_NAMES, PICKUP_LOCS, LOC_CHARS


def parse_state(s: str) -> tuple[int, int, int, int]:
    parts = s.split(',')
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "state must be 'x,y,P,D' with four integers separated by commas"
        )
    try:
        x, y, P, D = map(int, parts)
    except ValueError:
        raise argparse.ArgumentTypeError("state values must be integers")
    # validate ranges
    if not (0 <= x <= 4) or not (0 <= y <= 4):
        raise argparse.ArgumentTypeError("x and y must be in [0,4]")
    if not (0 <= P <= 4):
        raise argparse.ArgumentTypeError("P (passenger) must be in [0,4]")
    if not (0 <= D <= 3):
        raise argparse.ArgumentTypeError("D (destination) must be in [0,3]")
    return (x, y, P, D)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Taxi-v3 Robustness Region visualisation")
    parser.add_argument(
        "--model-path", type=Path,
        default=Path("data/experiments/models/Taxi-v3_DQN_model_20250423_173106/model.zip"),
        help="Path to SB3 .zip model file"
    )
    parser.add_argument(
        "--model-name", type=str, default="taxi_dqn",
        help="Identifier used to name output folder"
    )
    parser.add_argument(
        "--state", type=parse_state, default="1,1,1,2",
        help="Seed state as 'x,y,P,D'"
    )
    parser.add_argument(
        "--timestamp", type=str,
        help="Timestamp override (format YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--hide-walls", action="store_false", dest="show_walls",
        help="Do not draw walls on plots"
    )
    parser.set_defaults(show_walls=True)
    args = parser.parse_args(argv)

    # Prepare output directory
    ts = args.timestamp or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path.cwd() / "data" / "experiments" / "rr" / f"{args.model_name}_robustness_region" / f"time_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create environments
    base_env = gym.make("Taxi-v3")
    obs_env = OneHotObs(base_env)  # for model.predict

    # Load model
    model = DQN.load(str(args.model_path), env=obs_env)

    # Compute RR
    rr = compute_rr_taxi(args.state, model, base_env)
    tuples = sorted(rr["rr_tuple_set"])
    depths_map = rr["rr_depths"]

    # Save YAML
    yaml_data = {
        "metadata": {
            "model_name": args.model_name,
            "seed_tuple": list(args.state),
            "timestamp": ts,
            "initial_action": rr["initial_action"],
            **rr["stats"]
        },
        "rr_tuples": [list(t) for t in tuples],
        "rr_depths": [depths_map[t] for t in tuples]
    }
    yaml_path = out_dir / "robustness_region.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)
    print(f"Saved YAML → {yaml_path.relative_to(Path.cwd())}")

    # --- Initial state visualization ---
    # Create standalone grid showing initial taxi, passenger, destination, and action
    fig0, ax0 = plt.subplots(figsize=(6, 6))
    # Prepare grid: mask all but the initial state, color by initial action
    A0 = np.full((5, 5), -1, dtype=int)
    row0, col0 = args.state[0], args.state[1]
    A0[row0, col0] = rr["initial_action"]
    mask0 = (A0 == -1)
    im0 = ax0.imshow(np.ma.array(A0, mask=mask0), cmap=_COLORMAP, vmin=0, vmax=5)
    # Determine pickup and dest cell positions
    pickup0 = PICKUP_LOCS[args.state[2]] if args.state[2] < 4 else None
    dest0 = PICKUP_LOCS[args.state[3]]
    _annotate_grid(ax0, A0, pickup0, dest0, show_walls=args.show_walls)
    # Mark initial taxi location with 'S'
    ax0.text(col0, row0, 'S', ha='center', va='center', fontsize='x-large', color='red', weight='bold')
    # Add legend for action colors
    legend0 = [plt.Line2D([0], [0], marker="s", linestyle="", color=CB_PALETTE[a], label=ACTION_NAMES[a]) for a in range(len(ACTION_NAMES))]
    fig0.legend(handles=legend0, loc="lower center", ncol=6, title="Action", fontsize="small")
    fig0.tight_layout()
    init_path = out_dir / f"initial_state_{args.state[0]}_{args.state[1]}_{args.state[2]}_{args.state[3]}.png"
    fig0.savefig(init_path, dpi=150, bbox_inches="tight")
    plt.close(fig0)
    print(f"Saved initial-state image → {init_path.relative_to(Path.cwd())}")

    # Plot grid: 4x4 subplots
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for d in range(4):
        passenger_list = [p for p in range(5) if p != d][:3] + [4]
        for j, p in enumerate(passenger_list):
            ax = axes[d, j]
            A = np.full((5, 5), -1, dtype=int)
            for (x, y, P, D) in tuples:
                if P == p and D == d:
                    A[x, y] = rr["initial_action"]
            mask = (A == -1)
            ax.imshow(np.ma.array(A, mask=mask), cmap=_COLORMAP, vmin=0, vmax=5)
            pickup = PICKUP_LOCS[p] if p < 4 else None
            dest = PICKUP_LOCS[d]
            _annotate_grid(ax, A, pickup, dest, show_walls=args.show_walls)
            # Mark seed state with 'S' on the corresponding subplot
            if (d, p) == (args.state[3], args.state[2]):
                ax.text(args.state[1], args.state[0], 'S', ha='center', va='center', fontsize='x-large', color='red', weight='bold')
    # Shared legend
    legend_elems = [plt.Line2D([0], [0], marker="s", linestyle="", color=CB_PALETTE[a],
                    label=ACTION_NAMES[a]) for a in range(len(ACTION_NAMES))]
    fig.legend(handles=legend_elems, loc="lower center", ncol=6, title="Action")
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    # Save figure
    img_path = out_dir / f"robust_rr_seed_{args.state[0]}_{args.state[1]}_{args.state[2]}_{args.state[3]}.png"
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG → {img_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
