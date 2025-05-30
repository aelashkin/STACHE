#!/usr/bin/env python3
"""
CLI for computing and visualizing the robustness region (RR) of a Taxi-v3 policy.

Now expects --model-path to be a folder containing model.zip;
model_name is derived from that folder's name.
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

from stache.explainability.taxi.robust_taxi import compute_rr_taxi, translate_tuple_to_onehot
from stache.explainability.taxi.taxi_policy_map import OneHotObs, _annotate_grid, _COLORMAP, CB_PALETTE, ACTION_NAMES, PICKUP_LOCS, LOC_CHARS


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
        "--model-path",
        type=Path,
        default=Path("data/experiments/models/Taxi-v3_DQN_model_50"),
        help="Path to folder containing model.zip"
    )
    parser.add_argument(
        "--state", type=parse_state, default="0,0,0,2", # Default seed state
        help="Seed state as 'x,y,P,D'"
    )
    parser.add_argument(
        "--hide-walls", action="store_false", dest="show_walls",
        help="Do not draw walls on plots"
    )
    parser.set_defaults(show_walls=True)
    args = parser.parse_args(argv)

    # Validate model.zip and derive model_name
    zip_path = args.model_path / "model.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"Could not find model.zip in {args.model_path}")
    model_name = args.model_path.name

    # Prepare output directory
    # Ensure args.state is a tuple of ints for string formatting if not already
    s_tuple = args.state
    seed_str = f"{s_tuple[0]}_{s_tuple[1]}_{s_tuple[2]}_{s_tuple[3]}"
    out_dir = Path.cwd() / "data" / "experiments" / "rr" / "taxi_robustness_region" / model_name / seed_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create environments
    base_env = gym.make("Taxi-v3")
    # obs_env is used for model loading if the model expects a wrapped env.
    # However, DQN.load can often handle this if the wrapper is simple or if not strictly needed at load time.
    # For predict, we pass the observation from translate_tuple_to_onehot(base_env, ...)
    # For compute_rr_taxi, we pass base_env.
    model = DQN.load(str(zip_path), env=None) # Pass base_env or wrapped_env if model.load requires it

    # Compute RR and Counterfactuals
    # compute_rr_taxi expects the base_env for its 'env' parameter
    rr = compute_rr_taxi(s_tuple, model, base_env)
    tuples = sorted(rr["rr_tuple_set"])
    depths_map = rr["rr_depths"]
    s0_initial_action = rr["initial_action"]

    # Save YAML (existing logic)
    yaml_data = {
        "metadata": {
            "model_name": model_name,
            "seed_tuple": list(s_tuple),
            "initial_action": s0_initial_action,
            **rr["stats"]
        },
        "rr_tuples": [list(t) for t in tuples],
        "rr_depths": [depths_map[t] for t in tuples],
        "counterfactuals_found": [ # Store raw counterfactuals with depth
            {"state": list(cf_s), "depth": cf_d} for cf_s, cf_d in rr.get("counterfactuals", [])
        ]
    }
    yaml_path = out_dir / "robustness_region.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)
    print(f"Saved YAML → {yaml_path.relative_to(Path.cwd())}")

    # --- Initial state visualization ---
    # Create standalone grid showing initial taxi, passenger, destination, and action
    fig0, ax0 = plt.subplots(figsize=(6, 6))
    # Prepare grid: mask all but the initial state, color by initial action
    A0 = np.full((5, 5), -1, dtype=int)
    row0, col0 = s_tuple[0], s_tuple[1] # x, y from seed state
    A0[row0, col0] = s0_initial_action
    mask0 = (A0 == -1)
    im0 = ax0.imshow(np.ma.array(A0, mask=mask0), cmap=_COLORMAP, vmin=0, vmax=5)
    # Determine pickup and dest cell positions
    pickup0 = PICKUP_LOCS[s_tuple[2]] if s_tuple[2] < 4 else None
    dest0 = PICKUP_LOCS[s_tuple[3]]
    _annotate_grid(ax0, A0, pickup0, dest0, show_walls=args.show_walls)
    # Mark initial taxi location with 'S', offset if overlapping P or D
    state_coord = (row0, col0)
    text_x_s0, text_y_s0 = col0, row0
    ha_s0, va_s0 = 'center', 'center'
    if (pickup0 and state_coord == pickup0) or state_coord == dest0:
        text_x_s0 += 0.15 # Offset 'S'
        ha_s0 = 'left'
    ax0.text(text_x_s0, text_y_s0, 'S', ha=ha_s0, va=va_s0, fontsize='x-large', color='red', weight='bold')
    fig0.suptitle(f"Initial state {s_tuple} (Action: {ACTION_NAMES[s0_initial_action]})", fontsize="large", y=1.02)
    legend0_elems = [plt.Line2D([0], [0], marker="s", linestyle="", color=CB_PALETTE[a], label=ACTION_NAMES[a]) for a in range(len(ACTION_NAMES))]
    fig0.legend(handles=legend0_elems, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, title="Action", fontsize="small") # Adjusted bbox & ncol
    fig0.tight_layout(rect=[0, 0.05, 1, 1])
    init_path = out_dir / f"initial_state_{seed_str}.png"
    fig0.savefig(init_path, dpi=150, bbox_inches="tight")
    plt.close(fig0)
    print(f"Saved initial-state image → {init_path.relative_to(Path.cwd())}")

    # --- Robustness Region visualization ---
    dest_order = [3, 1, 0, 2]  # B, G, R, Y
    fig, axes = plt.subplots(4, 4, figsize=(14, 14)) # Consider adjusting figsize if needed
    for row_idx, d_plot_rr in enumerate(dest_order):
        passenger_list_rr = [p_rr for p_rr in range(5) if p_rr != d_plot_rr][:3] + [4]
        for col_idx, p_plot_rr in enumerate(passenger_list_rr):
            ax_rr = axes[row_idx, col_idx]
            A_rr = np.full((5, 5), -1, dtype=int)
            for (x_rr, y_rr, P_rr, D_rr) in tuples: # These are states in RR
                if P_rr == p_plot_rr and D_rr == d_plot_rr:
                    A_rr[x_rr, y_rr] = s0_initial_action # All states in RR take this action
            mask_rr = (A_rr == -1)
            ax_rr.imshow(np.ma.array(A_rr, mask=mask_rr), cmap=_COLORMAP, vmin=0, vmax=5)
            pickup_rr = PICKUP_LOCS[p_plot_rr] if p_plot_rr < 4 else None
            dest_rr_loc = PICKUP_LOCS[d_plot_rr]
            _annotate_grid(ax_rr, A_rr, pickup_rr, dest_rr_loc, show_walls=args.show_walls)
            # Mark seed state 'S' on the corresponding subplot
            if (d_plot_rr, p_plot_rr) == (s_tuple[3], s_tuple[2]): # If current subplot matches seed's D, P
                s0_x, s0_y = s_tuple[0], s_tuple[1]
                text_x_s_rr, text_y_s_rr = s0_y, s0_x
                ha_s_rr, va_s_rr = 'center', 'center'
                if (pickup_rr and (s0_x, s0_y) == pickup_rr) or (s0_x, s0_y) == dest_rr_loc:
                    text_x_s_rr += 0.15
                    ha_s_rr = 'left'
                ax_rr.text(text_x_s_rr, text_y_s_rr, 'S', ha=ha_s_rr, va=va_s_rr, fontsize='x-large', color='red', weight='bold')
            # Titles for subplots
            pass_char_rr = LOC_CHARS.get(p_plot_rr, 'InTaxi') if p_plot_rr == 4 else LOC_CHARS[p_plot_rr]
            ax_rr.set_title(f"P={pass_char_rr}, D={LOC_CHARS[d_plot_rr]}", fontsize='medium')

    fig.suptitle(f"Robustness Region for s0={s_tuple} (Action: {ACTION_NAMES[s0_initial_action]})", y=0.98, fontsize="large") # Adjusted y
    # Shared legend for RR plot
    legend_elems_rr = [plt.Line2D([0], [0], marker="s", linestyle="", color=CB_PALETTE[a], label=ACTION_NAMES[a]) for a in range(len(ACTION_NAMES))]
    fig.legend(handles=legend_elems_rr, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=6, title="Action in Region", fontsize="small") # Adjusted bbox
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect to make space for suptitle and legend
    rr_path = out_dir / f"robustness_region_{seed_str}.png"
    fig.savefig(rr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved robustness region image → {rr_path.relative_to(Path.cwd())}")


    # --- NEW: Minimal Counterfactuals Visualization ---
    all_counterfactuals_with_depth = rr.get("counterfactuals", [])
    minimal_counterfactuals_for_plot = []

    if all_counterfactuals_with_depth:
        min_depth = min(d for s, d in all_counterfactuals_with_depth)
        minimal_cf_states_at_min_depth = [s for s, d in all_counterfactuals_with_depth if d == min_depth]
        
        for mcf_state_tuple in minimal_cf_states_at_min_depth:
            # We need the action for this mcf_state
            mcf_vec = translate_tuple_to_onehot(base_env, mcf_state_tuple)
            mcf_action_arr, _ = model.predict(mcf_vec, deterministic=True)
            mcf_action = int(mcf_action_arr[0]) if isinstance(mcf_action_arr, (np.ndarray, list, tuple)) else int(mcf_action_arr)
            # Only include if the action is actually different (it should be by definition from robust_taxi.py)
            if mcf_action != s0_initial_action:
                 minimal_counterfactuals_for_plot.append((mcf_state_tuple, mcf_action))
    
    if minimal_counterfactuals_for_plot:
        fig_cf, axes_cf = plt.subplots(4, 4, figsize=(14, 14)) # Same layout as RR
        
        s0_x, s0_y, s0_P, s0_D = s_tuple

        for row_idx, d_plot_cf in enumerate(dest_order): # Same dest_order
            passenger_list_cf = [p_cf for p_cf in range(5) if p_cf != d_plot_cf][:3] + [4]
            for col_idx, p_plot_cf in enumerate(passenger_list_cf):
                ax_cf = axes_cf[row_idx, col_idx]
                A_cf = np.full((5, 5), -1, dtype=int) # Grid for actions

                # Plot minimal counterfactual states first
                for mcf_s, mcf_a in minimal_counterfactuals_for_plot:
                    mcf_x, mcf_y, mcf_P, mcf_D = mcf_s
                    if mcf_D == d_plot_cf and mcf_P == p_plot_cf:
                        A_cf[mcf_x, mcf_y] = mcf_a
                
                # Plot initial state s0 on top, if it belongs to this subplot
                if s0_D == d_plot_cf and s0_P == p_plot_cf:
                    A_cf[s0_x, s0_y] = s0_initial_action

                mask_cf = (A_cf == -1)
                ax_cf.imshow(np.ma.array(A_cf, mask=mask_cf), cmap=_COLORMAP, vmin=0, vmax=5)
                
                pickup_cf = PICKUP_LOCS[p_plot_cf] if p_plot_cf < 4 else None
                dest_cf_loc = PICKUP_LOCS[d_plot_cf]
                _annotate_grid(ax_cf, A_cf, pickup_cf, dest_cf_loc, show_walls=args.show_walls)

                # Mark initial state s0 with 'S' if it's in this subplot
                if (s0_D, s0_P) == (d_plot_cf, p_plot_cf):
                    text_x_s_cf, text_y_s_cf = s0_y, s0_x
                    ha_s_cf, va_s_cf = 'center', 'center'
                    if (pickup_cf and (s0_x, s0_y) == pickup_cf) or \
                       ((s0_x, s0_y) == dest_cf_loc):
                        text_x_s_cf += 0.15
                        ha_s_cf = 'left'
                    ax_cf.text(text_x_s_cf, text_y_s_cf, 'S', ha=ha_s_cf, va=va_s_cf, fontsize='x-large', color='red', weight='bold')
                
                pass_char_cf = LOC_CHARS.get(p_plot_cf, 'InTaxi') if p_plot_cf == 4 else LOC_CHARS[p_plot_cf]
                ax_cf.set_title(f"P={pass_char_cf}, D={LOC_CHARS[d_plot_cf]}", fontsize='medium')

        fig_cf.suptitle(f"Minimal Counterfactuals for s0={s_tuple} (s0 Action: {ACTION_NAMES[s0_initial_action]})", y=0.98, fontsize="large")
        
        legend_elems_cf = [plt.Line2D([0], [0], marker="s", linestyle="", color=CB_PALETTE[a], label=ACTION_NAMES[a]) for a in range(len(ACTION_NAMES))]
        fig_cf.legend(handles=legend_elems_cf, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=6, title="Action", fontsize="small")
        
        fig_cf.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        cf_image_path = out_dir / f"counterfactuals_seed_{seed_str}.png"
        fig_cf.savefig(cf_image_path, dpi=150, bbox_inches="tight")
        plt.close(fig_cf)
        print(f"Saved minimal counterfactuals image → {cf_image_path.relative_to(Path.cwd())}")
    else:
        print(f"No minimal counterfactuals found for seed {s_tuple} to visualize.")

if __name__ == "__main__":
    main()
