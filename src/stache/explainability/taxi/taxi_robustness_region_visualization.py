#!/usr/bin/env python3
"""
CLI for computing and visualizing the robustness region (RR) of a Taxi-v3 policy.

Now expects --model-path to be a folder containing model.zip;
model_name is derived from that folder's name.
"""
import argparse
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stache.explainability.artifacts import save_result
from stache.explainability.connectors.taxi import TaxiConnector
from stache.explainability.core.models import SearchResult
from stache.explainability.taxi.model_loading import load_trusted_taxi_model
from stache.explainability.taxi.robust_taxi import compute_taxi_rr
from stache.explainability.taxi.taxi_policy_map import (
    ACTION_NAMES,
    CB_PALETTE,
    LOC_CHARS,
    PICKUP_LOCS,
    _annotate_grid,
    _COLORMAP,
    _policy_map_panel_pairs,
)


_DESTINATION_DISPLAY_ORDER = (3, 1, 0, 2)  # B, G, R, Y


def _taxi_panel_pairs(
    destination_order: Iterable[int] = _DESTINATION_DISPLAY_ORDER,
) -> tuple[tuple[int, int], ...]:
    """Return all thesis-universe passenger/destination panel pairs.

    Passenger factors include every waiting location (also ``P == D``) and the
    in-taxi value 4.  Both Taxi visualizers share this full 20-panel view.
    """

    return _policy_map_panel_pairs(destination_order)


def _minimal_counterfactuals_for_plot(
    result: SearchResult[object, object],
) -> tuple[tuple[object, int], ...]:
    """Project cached result actions; visualization never queries the policy."""

    return tuple(
        (record.state, record.action)
        for record in result.minimal_counterfactuals
    )


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
        "--state", type=parse_state, default=(0, 0, 0, 2),
        help="Seed state as 'x,y,P,D'"
    )
    parser.add_argument(
        "--acknowledge-trusted-model",
        action="store_true",
        help="Confirm that model.zip came from a trusted source",
    )
    parser.add_argument(
        "--hide-walls", action="store_false", dest="show_walls",
        help="Do not draw walls on plots"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing canonical RR artifact",
    )
    parser.set_defaults(show_walls=True)
    args = parser.parse_args(argv)
    if not args.acknowledge_trusted_model:
        parser.error(
            "--acknowledge-trusted-model is required before loading model.zip"
        )

    zip_path = args.model_path / "model.zip"
    model_name = args.model_path.name
    loaded_model = load_trusted_taxi_model(
        zip_path,
        acknowledge_trusted_model=args.acknowledge_trusted_model,
    )

    # Prepare output directory
    # Ensure args.state is a tuple of ints for string formatting if not already
    s_tuple = args.state
    seed_str = f"{s_tuple[0]}_{s_tuple[1]}_{s_tuple[2]}_{s_tuple[3]}"
    out_dir = Path.cwd() / "data" / "experiments" / "rr" / "taxi_robustness_region" / model_name / seed_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute RR and Counterfactuals
    rr = compute_taxi_rr(
        s_tuple,
        model=loaded_model.model,
        model_fingerprint=loaded_model.model_fingerprint,
        model_manifest=loaded_model.manifest,
    )
    tuples = [record.state for record in rr.region]
    s0_initial_action = rr.seed_action

    artifact_path = out_dir / "robustness_region.yaml"
    save_result(
        artifact_path,
        rr,
        TaxiConnector(),
        overwrite=args.overwrite,
    )
    print(f"Saved RR artifact → {artifact_path.relative_to(Path.cwd())}")

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
    dest_order = _DESTINATION_DISPLAY_ORDER
    fig, axes = plt.subplots(4, 5, figsize=(17.5, 14))
    panel_pairs = _taxi_panel_pairs(dest_order)
    for panel_index, (p_plot_rr, d_plot_rr) in enumerate(panel_pairs):
        row_idx, col_idx = divmod(panel_index, 5)
        ax_rr = axes[row_idx, col_idx]
        action_grid = np.full((5, 5), -1, dtype=int)
        for taxi_row, taxi_column, passenger, destination in tuples:
            if passenger == p_plot_rr and destination == d_plot_rr:
                action_grid[taxi_row, taxi_column] = s0_initial_action
        mask_rr = action_grid == -1
        ax_rr.imshow(
            np.ma.array(action_grid, mask=mask_rr),
            cmap=_COLORMAP,
            vmin=0,
            vmax=5,
        )
        pickup_rr = PICKUP_LOCS[p_plot_rr] if p_plot_rr < 4 else None
        dest_rr_loc = PICKUP_LOCS[d_plot_rr]
        _annotate_grid(
            ax_rr,
            action_grid,
            pickup_rr,
            dest_rr_loc,
            show_walls=args.show_walls,
        )
        if (d_plot_rr, p_plot_rr) == (s_tuple[3], s_tuple[2]):
            s0_x, s0_y = s_tuple[0], s_tuple[1]
            text_x_s_rr, text_y_s_rr = s0_y, s0_x
            horizontal_alignment = "center"
            if (
                pickup_rr and (s0_x, s0_y) == pickup_rr
            ) or (s0_x, s0_y) == dest_rr_loc:
                text_x_s_rr += 0.15
                horizontal_alignment = "left"
            ax_rr.text(
                text_x_s_rr,
                text_y_s_rr,
                "S",
                ha=horizontal_alignment,
                va="center",
                fontsize="x-large",
                color="red",
                weight="bold",
            )
        passenger_label = (
            LOC_CHARS[p_plot_rr] if p_plot_rr < 4 else "InTaxi"
        )
        ax_rr.set_title(
            f"P={passenger_label}, D={LOC_CHARS[d_plot_rr]}",
            fontsize="medium",
        )

    fig.suptitle(f"Robustness Region for s0={s_tuple} (Action: {ACTION_NAMES[s0_initial_action]})", y=0.98, fontsize="large") # Adjusted y
    # Shared legend for RR plot
    legend_elems_rr = [plt.Line2D([0], [0], marker="s", linestyle="", color=CB_PALETTE[a], label=ACTION_NAMES[a]) for a in range(len(ACTION_NAMES))]
    fig.legend(handles=legend_elems_rr, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=6, title="Action in Region", fontsize="small") # Adjusted bbox
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect to make space for suptitle and legend
    rr_path = out_dir / f"robustness_region_{seed_str}.png"
    fig.savefig(rr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved robustness region image → {rr_path.relative_to(Path.cwd())}")
    # --- Minimal Counterfactuals Visualization ---
    minimal_counterfactuals_for_plot = _minimal_counterfactuals_for_plot(rr)

    if minimal_counterfactuals_for_plot:
        fig_cf, axes_cf = plt.subplots(4, 5, figsize=(17.5, 14))
        s0_x, s0_y, s0_P, s0_D = s_tuple

        for panel_index, (p_plot_cf, d_plot_cf) in enumerate(panel_pairs):
            row_idx, col_idx = divmod(panel_index, 5)
            ax_cf = axes_cf[row_idx, col_idx]
            action_grid = np.full((5, 5), -1, dtype=int)

            for mcf_state, mcf_action in minimal_counterfactuals_for_plot:
                mcf_x, mcf_y, mcf_passenger, mcf_destination = mcf_state
                if (
                    mcf_destination == d_plot_cf
                    and mcf_passenger == p_plot_cf
                ):
                    action_grid[mcf_x, mcf_y] = mcf_action

            if s0_D == d_plot_cf and s0_P == p_plot_cf:
                action_grid[s0_x, s0_y] = s0_initial_action

            mask_cf = action_grid == -1
            ax_cf.imshow(
                np.ma.array(action_grid, mask=mask_cf),
                cmap=_COLORMAP,
                vmin=0,
                vmax=5,
            )

            pickup_cf = PICKUP_LOCS[p_plot_cf] if p_plot_cf < 4 else None
            dest_cf_loc = PICKUP_LOCS[d_plot_cf]
            _annotate_grid(
                ax_cf,
                action_grid,
                pickup_cf,
                dest_cf_loc,
                show_walls=args.show_walls,
            )

            if (s0_D, s0_P) == (d_plot_cf, p_plot_cf):
                text_x_s_cf, text_y_s_cf = s0_y, s0_x
                horizontal_alignment = "center"
                if (
                    pickup_cf and (s0_x, s0_y) == pickup_cf
                ) or (s0_x, s0_y) == dest_cf_loc:
                    text_x_s_cf += 0.15
                    horizontal_alignment = "left"
                ax_cf.text(
                    text_x_s_cf,
                    text_y_s_cf,
                    "S",
                    ha=horizontal_alignment,
                    va="center",
                    fontsize="x-large",
                    color="red",
                    weight="bold",
                )

            passenger_label = (
                LOC_CHARS[p_plot_cf] if p_plot_cf < 4 else "InTaxi"
            )
            ax_cf.set_title(
                f"P={passenger_label}, D={LOC_CHARS[d_plot_cf]}",
                fontsize="medium",
            )

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
