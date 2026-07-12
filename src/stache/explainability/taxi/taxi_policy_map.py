#!/usr/bin/env python3
"""Render a Taxi policy over the thesis-compatible 500-state universe.

The policy map queries every factored state declared by :class:`TaxiConnector`
and renders all 20 passenger/destination configurations, including ``P == D``.
The connector owns state indexing, one-hot observations, and action metadata;
this module contains only model loading, primitive YAML output, and rendering.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import datetime as dt
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap

from stache.explainability.connectors.taxi import TaxiConnector
from stache.explainability.core.policy import (
    ModelActionOracle,
    ModelManifest,
    normalize_discrete_action,
)
from stache.explainability.taxi.model_loading import load_trusted_taxi_model


_ACTION_METADATA = TaxiConnector().action_metadata
ACTION_NAMES = {
    int(item["action"]): str(item["label"]).title()
    for item in _ACTION_METADATA
}

CB_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish-green
    "#CC79A7",  # reddish-purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
]

_COLORMAP = ListedColormap(CB_PALETTE, name="taxi_actions")

# Rendering-only map locations as (row, column).
PICKUP_LOCS = {
    0: (0, 0),  # R
    1: (0, 4),  # G
    2: (4, 0),  # Y
    3: (4, 3),  # B
}
LOC_CHARS = {0: "R", 1: "G", 2: "Y", 3: "B"}
_DESTINATION_DISPLAY_ORDER = (3, 1, 0, 2)  # B, G, R, Y


def _policy_map_panel_pairs(
    destination_order: Iterable[int] = _DESTINATION_DISPLAY_ORDER,
) -> tuple[tuple[int, int], ...]:
    """Return all 20 ``(passenger, destination)`` rendering panels."""

    return tuple(
        (passenger, destination)
        for destination in destination_order
        for passenger in range(5)
    )


def collect_state_actions(
    model: object,
    env: object | None = None,
    base_env: object | None = None,
    *,
    connector: TaxiConnector | None = None,
    model_fingerprint: str | None = None,
    model_manifest: ModelManifest | None = None,
) -> dict[int, int]:
    """Query ``model`` exactly once for each of the 500 declared Taxi states.

    The historical ``env`` and ``base_env`` positional arguments are accepted
    with a deprecation warning but no longer provide encoding or enumeration.
    """

    connector = connector or TaxiConnector()
    if env is not None or base_env is not None:
        warnings.warn(
            "collect_state_actions env/base_env arguments are deprecated; "
            "TaxiConnector now owns enumeration and observation encoding",
            DeprecationWarning,
            stacklevel=2,
        )
        _validate_legacy_policy_envs(env, base_env)
    if not isinstance(model_manifest, ModelManifest):
        model_manifest = getattr(model, "stache_model_manifest", None)
    if not isinstance(model_manifest, ModelManifest):
        raise ValueError("model_manifest is required for policy-map collection")
    if model_fingerprint is None:
        model_fingerprint = model_manifest.model_fingerprint
    if not isinstance(model_fingerprint, str) or not model_fingerprint.strip():
        raise ValueError("model_fingerprint is required for policy-map collection")
    oracle = ModelActionOracle(
        connector,
        model,
        source_fingerprint=model_fingerprint,
        manifest=model_manifest,
    )
    mapping: dict[int, int] = {}
    for state in connector.declared_states():
        lookup_key = connector.policy_lookup_key(state)
        if lookup_key in mapping:
            raise RuntimeError(
                f"TaxiConnector produced duplicate policy key {lookup_key}"
            )
        mapping[lookup_key] = oracle.action(state)

    if set(mapping) != set(range(500)):
        raise RuntimeError("Taxi policy map must cover exactly keys 0..499")
    return mapping


def save_mapping_yaml(mapping: Mapping[int, object], filepath: Path) -> None:
    """Save a complete primitive-only 500-state policy mapping as safe YAML."""

    normalized = _validated_mapping(mapping)
    with filepath.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(normalized, stream, sort_keys=True)


def build_action_grid(
    mapping: Mapping[int, object] | object,
    passenger_loc: int | Mapping[int, object],
    dest_idx: int,
    legacy_dest_idx: int | None = None,
    *,
    connector: TaxiConnector | None = None,
) -> np.ndarray:
    """Build one 5x5 action grid, accepting the deprecated env-first call."""

    connector = connector or TaxiConnector()
    if legacy_dest_idx is not None:
        warnings.warn(
            "build_action_grid(taxi_env, mapping, passenger, destination) is "
            "deprecated; pass mapping, passenger, destination instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mapping, passenger_loc, dest_idx = (
            passenger_loc,
            dest_idx,
            legacy_dest_idx,
        )
    if not isinstance(mapping, Mapping):
        raise TypeError("Taxi policy mapping must be a mapping")
    if type(passenger_loc) is not int:
        raise TypeError("Taxi passenger factor must be an integer")
    connector.validate_state((0, 0, passenger_loc, dest_idx))
    grid = np.full((5, 5), fill_value=-1, dtype=int)
    for row in range(5):
        for column in range(5):
            state = (row, column, passenger_loc, dest_idx)
            lookup_key = connector.policy_lookup_key(state)
            try:
                raw_action = mapping[lookup_key]
            except KeyError as error:
                raise KeyError(
                    "Taxi policy mapping has no action for state "
                    f"{state} (key {lookup_key})"
                ) from error
            grid[row, column] = normalize_discrete_action(
                raw_action,
                connector.action_spec.count,
            )
    return grid


def _annotate_grid(
    ax: object,
    grid: np.ndarray,
    passenger: tuple[int, int] | None,
    dest: tuple[int, int],
    show_walls: bool = True,
) -> None:
    """Overlay passenger/destination labels and Taxi road walls."""

    ax.set_xticks([])  # type: ignore[attr-defined]
    ax.set_yticks([])  # type: ignore[attr-defined]
    for row in range(5):
        for column in range(5):
            label = ""
            if passenger and (row, column) == passenger:
                label = "P"
            if (row, column) == dest:
                label = "D" if not label else "PD"
            if label:
                ax.text(  # type: ignore[attr-defined]
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize="medium",
                    color="black",
                    weight="bold",
                )
    ax.set_xlim(-0.5, 4.5)  # type: ignore[attr-defined]
    ax.set_ylim(4.5, -0.5)  # type: ignore[attr-defined]

    if show_walls:
        wall_kwargs = {"color": "black", "linewidth": 2.5}
        ax.plot([0.5, 0.5], [2.5, 4.5], **wall_kwargs)  # type: ignore[attr-defined]
        ax.plot([1.5, 1.5], [-0.5, 1.5], **wall_kwargs)  # type: ignore[attr-defined]
        ax.plot([3.5, 3.5], [2.5, 4.5], **wall_kwargs)  # type: ignore[attr-defined]


def plot_dest_maps(
    mapping: Mapping[int, object] | object,
    dest_idx: int | Mapping[int, object],
    output_path: Path | int,
    legacy_output_path: Path | None = None,
    show_walls: bool = True,
    *,
    connector: TaxiConnector | None = None,
) -> None:
    """Render five panels, accepting the deprecated env-first call shape."""

    connector = connector or TaxiConnector()
    if legacy_output_path is not None:
        warnings.warn(
            "plot_dest_maps(taxi_env, mapping, destination, output) is "
            "deprecated; pass mapping, destination, output instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mapping, dest_idx, output_path = (
            dest_idx,
            output_path,
            legacy_output_path,
        )
    if not isinstance(mapping, Mapping):
        raise TypeError("Taxi policy mapping must be a mapping")
    if type(dest_idx) is not int:
        raise TypeError("Taxi destination factor must be an integer")
    if not isinstance(output_path, Path):
        raise TypeError("Taxi policy-map output_path must be a pathlib.Path")
    connector.validate_state((0, 0, 0, dest_idx))
    passenger_configs = tuple(range(5))
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    for ax, passenger in zip(axes, passenger_configs, strict=True):
        grid = build_action_grid(
            mapping,
            passenger,
            dest_idx,
            connector=connector,
        )
        ax.imshow(grid, cmap=_COLORMAP, vmin=0, vmax=5)
        pickup_cell = PICKUP_LOCS[passenger] if passenger < 4 else None
        destination_cell = PICKUP_LOCS[dest_idx]
        _annotate_grid(
            ax,
            grid,
            pickup_cell,
            destination_cell,
            show_walls=show_walls,
        )
        subtitle = (
            f"Passenger at {LOC_CHARS[passenger]}"
            if passenger < 4
            else "Passenger in taxi"
        )
        ax.set_title(subtitle, fontsize="small")

    fig.legend(
        handles=_legend_elements(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=6,
        title="Action taken by policy",
        fontsize="small",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.suptitle(f"Destination = {LOC_CHARS[dest_idx]}", y=1.05)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_visualisation(
    model_path: Path,
    timestamp: str | None = None,
    show_walls: bool = True,
    *,
    acknowledge_trusted_model: bool = False,
    overwrite: bool = False,
) -> None:
    """Load a DQN and write the 500-state mapping and 20-panel images."""

    model_name = model_path.name
    zip_path = model_path / "model.zip"
    loaded_model = load_trusted_taxi_model(
        zip_path,
        acknowledge_trusted_model=acknowledge_trusted_model,
    )
    timestamp = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = (
        Path.cwd()
        / "data"
        / "experiments"
        / "rr"
        / "policy_map"
        / model_name
        / timestamp
    )
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"policy-map output already exists: {output_dir}; pass overwrite=True"
        )
    output_dir.mkdir(parents=True, exist_ok=overwrite)

    connector = TaxiConnector()
    mapping = collect_state_actions(
        loaded_model.model,
        connector=connector,
        model_fingerprint=loaded_model.model_fingerprint,
        model_manifest=loaded_model.manifest,
    )

    yaml_path = output_dir / "state_action_mapping.yaml"
    save_mapping_yaml(mapping, yaml_path)
    print(f"Saved mapping -> {yaml_path.relative_to(Path.cwd())}")

    for destination in range(4):
        image_path = output_dir / f"policy_map_dest_{LOC_CHARS[destination]}.png"
        plot_dest_maps(
            mapping,
            destination,
            image_path,
            connector=connector,
            show_walls=show_walls,
        )
        print(f"Saved visualisation -> {image_path.relative_to(Path.cwd())}")

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for panel_index, (passenger, destination) in enumerate(
        _policy_map_panel_pairs()
    ):
        row_index, column_index = divmod(panel_index, 5)
        ax = axes[row_index, column_index]
        grid = build_action_grid(
            mapping,
            passenger,
            destination,
            connector=connector,
        )
        ax.imshow(grid, cmap=_COLORMAP, vmin=0, vmax=5)
        pickup = PICKUP_LOCS[passenger] if passenger < 4 else None
        destination_cell = PICKUP_LOCS[destination]
        _annotate_grid(
            ax,
            grid,
            pickup,
            destination_cell,
            show_walls=show_walls,
        )
        passenger_label = (
            LOC_CHARS[passenger] if passenger < 4 else "InTaxi"
        )
        ax.set_title(
            f"P={passenger_label}, D={LOC_CHARS[destination]}",
            fontsize="small",
        )

    fig.legend(
        handles=_legend_elements(),
        loc="lower center",
        ncol=6,
        title="Action",
        fontsize="small",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.suptitle(f"500-state Policy Map for {model_name}", y=1.02)
    combined_path = output_dir / "policy_map.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined 4x5 visualisation -> {combined_path.relative_to(Path.cwd())}")


def _legend_elements() -> list[object]:
    return [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=CB_PALETTE[action],
            label=ACTION_NAMES[action],
        )
        for action in sorted(ACTION_NAMES)
    ]


def _validated_mapping(mapping: Mapping[int, object]) -> dict[int, int]:
    if not isinstance(mapping, Mapping):
        raise TypeError("Taxi policy mapping must be a mapping")
    if set(mapping) != set(range(500)):
        raise ValueError("Taxi policy mapping must contain exactly keys 0..499")
    connector = TaxiConnector()
    return {
        index: normalize_discrete_action(
            mapping[index],
            connector.action_spec.count,
        )
        for index in range(500)
    }


def _validate_legacy_policy_envs(
    env: object | None,
    base_env: object | None,
) -> None:
    observation_space = getattr(env, "observation_space", None)
    shape = getattr(observation_space, "shape", None)
    if shape is not None and tuple(shape) != (500,):
        raise ValueError(
            "legacy wrapped Taxi environment must expose shape (500,), "
            f"got {tuple(shape)!r}"
        )
    unwrapped = getattr(base_env, "unwrapped", base_env)
    state_space = getattr(unwrapped, "observation_space", None)
    state_count = getattr(state_space, "n", None)
    if state_count is not None and state_count != 500:
        raise ValueError(
            "legacy base Taxi environment must expose Discrete(500), "
            f"got n={state_count!r}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Taxi-v3 500-state thesis policy visualiser"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/experiments/models/Taxi-v3_DQN_model_50"),
        help="Path to the folder containing model.zip.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Use a fixed timestamp instead of the current datetime.",
    )
    parser.add_argument(
        "--acknowledge-trusted-model",
        action="store_true",
        help="Confirm that model.zip came from a trusted source.",
    )
    parser.add_argument(
        "--hide-walls",
        action="store_false",
        dest="show_walls",
        help="Do not draw the environment walls on the plots.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files in an existing fixed-timestamp output directory.",
    )
    parser.set_defaults(show_walls=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    if not args.acknowledge_trusted_model:
        raise SystemExit(
            "stache-viz-policy-map: error: --acknowledge-trusted-model is "
            "required before loading model.zip"
        )
    run_visualisation(
        args.model_path,
        args.timestamp,
        show_walls=args.show_walls,
        acknowledge_trusted_model=args.acknowledge_trusted_model,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
