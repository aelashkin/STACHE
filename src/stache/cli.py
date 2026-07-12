"""Installed command-line interface for reproducible STACHE computations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import yaml

from stache.utils.safe_yaml import safe_load_unique


class CliUsageError(ValueError):
    """A command configuration is invalid before computation starts."""


class CliExecutionError(RuntimeError):
    """An input was valid but its requested computation could not complete."""


_CONFIG_KEYS = {
    "domain",
    "state_universe",
    "seed",
    "policy_table",
    "model",
    "model_manifest",
    "acknowledge_trusted_model",
    "minimum_basis",
    "counterfactuals",
    "extent",
    "max_expanded",
    "max_policy_queries",
    "max_graph_depth",
    "output",
    "overwrite",
}


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = _nonnegative_integer(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stache",
        description="State-action transparency and counterfactual explanations.",
    )
    subcommands = parser.add_subparsers(dest="command")
    compute = subcommands.add_parser(
        "compute-rr",
        help="compute a robustness region and counterfactuals",
        description=(
            "Compute Taxi robustness-region and counterfactual results using "
            "the domain-neutral search core."
        ),
    )
    compute.add_argument(
        "--domain",
        choices=("taxi",),
        default=None,
        help="connector domain (Phase 1 supports taxi only)",
    )
    compute.add_argument(
        "--state-universe",
        choices=("taxi-factored-500",),
        default=None,
        help="declared connector state universe (all 500 factored Taxi states)",
    )
    compute.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INDEX",
        help="Taxi-v3 encoded seed index in [0, 499]",
    )
    policy = compute.add_mutually_exclusive_group()
    policy.add_argument(
        "--policy-table",
        type=Path,
        default=None,
        metavar="PATH",
        help="strict JSON/YAML mapping from Taxi indices to actions",
    )
    policy.add_argument(
        "--model",
        type=Path,
        default=None,
        metavar="PATH",
        help="Stable-Baselines3 DQN model archive",
    )
    compute.add_argument(
        "--model-manifest",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "versioned semantic manifest for --model "
            "(defaults to model.manifest.yaml beside the archive)"
        ),
    )
    compute.add_argument(
        "--acknowledge-trusted-model",
        action="store_true",
        default=None,
        help=(
            "confirm that --model came from a trusted source; Stable-Baselines3 "
            "archives deserialize Python objects, so this flag (or "
            "acknowledge_trusted_model: true in --config) is required before "
            "STACHE reads or loads the archive"
        ),
    )
    compute.add_argument(
        "--minimum-basis",
        choices=("graph_boundary", "formal_global"),
        default=None,
        help="basis used to certify minimal counterfactuals",
    )
    compute.add_argument(
        "--counterfactuals",
        choices=("minimal", "boundary", "both"),
        default=None,
        help="counterfactual projection to include",
    )
    compute.add_argument(
        "--extent",
        choices=("exact", "through_minimal_cf"),
        default=None,
        help="requested scientific search extent",
    )
    compute.add_argument(
        "--max-expanded",
        type=_nonnegative_integer,
        default=None,
        metavar="N",
        help="optional total expansion ceiling",
    )
    compute.add_argument(
        "--max-policy-queries",
        type=_positive_integer,
        default=None,
        metavar="N",
        help="optional total uncached policy-query ceiling",
    )
    compute.add_argument(
        "--max-graph-depth",
        type=_nonnegative_integer,
        default=None,
        metavar="N",
        help="optional graph-depth ceiling",
    )
    compute.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="safe YAML/JSON command configuration (CLI values take precedence)",
    )
    compute.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="versioned YAML result artifact",
    )
    compute.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="replace an existing output artifact atomically",
    )
    return parser


def _safe_mapping(
    path: Path,
    *,
    label: str,
    require_string_keys: bool = True,
) -> dict[Any, Any]:
    if not path.is_file():
        raise CliUsageError(f"{label} file does not exist or is not regular: {path}")
    try:
        value = safe_load_unique(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CliUsageError(f"cannot safely load {label} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise CliUsageError(f"{label} must contain a mapping")
    if require_string_keys and any(type(key) is not str for key in value):
        raise CliUsageError(f"{label} keys must be strings")
    return dict(value)


def _load_config(path: Path | None) -> tuple[dict[str, Any], Path | None]:
    if path is None:
        return {}, None
    resolved = path.expanduser().resolve()
    document = _safe_mapping(resolved, label="--config")
    if "compute_rr" in document:
        if set(document) != {"compute_rr"}:
            raise CliUsageError(
                "--config using a compute_rr section may not contain other root keys"
            )
        section = document["compute_rr"]
        if not isinstance(section, Mapping) or any(
            type(key) is not str for key in section
        ):
            raise CliUsageError("--config compute_rr must be a string-keyed mapping")
        document = dict(section)
    unknown = sorted(set(document) - _CONFIG_KEYS)
    if unknown:
        raise CliUsageError(
            "--config contains unknown compute-rr keys: " + ", ".join(unknown)
        )
    return document, resolved.parent


def _configured(
    arguments: argparse.Namespace,
    config: Mapping[str, Any],
    name: str,
    *,
    default: Any = None,
) -> Any:
    cli_value = getattr(arguments, name)
    if cli_value is not None:
        return cli_value
    return config.get(name, default)


def _path_value(value: object, *, option: str, config_dir: Path | None) -> Path:
    if not isinstance(value, (str, Path)):
        raise CliUsageError(f"{option} must be a filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute() and config_dir is not None:
        path = config_dir / path
    return Path(os.path.abspath(path))


def _exact_integer(value: object, *, option: str, minimum: int) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < minimum:
        raise CliUsageError(f"{option} must be an integer >= {minimum}")
    return value


def _exact_string_choice(
    value: object,
    *,
    option: str,
    choices: tuple[str, ...],
) -> str:
    if type(value) is not str or value not in choices:
        allowed = ", ".join(repr(choice) for choice in choices)
        raise CliUsageError(f"{option} must be one of: {allowed}")
    return value


def _validated_compute_config(arguments: argparse.Namespace) -> dict[str, Any]:
    config, config_dir = _load_config(arguments.config)
    domain = _exact_string_choice(
        _configured(arguments, config, "domain"),
        option="--domain",
        choices=("taxi",),
    )
    state_universe = _exact_string_choice(
        _configured(arguments, config, "state_universe"),
        option="--state-universe",
        choices=("taxi-factored-500",),
    )

    seed = _exact_integer(
        _configured(arguments, config, "seed"),
        option="--seed",
        minimum=0,
    )
    if seed is None or seed > 499:
        raise CliUsageError("--seed must be an integer in [0, 499]")

    cli_source_supplied = (
        arguments.policy_table is not None or arguments.model is not None
    )
    if cli_source_supplied:
        policy_table_value = arguments.policy_table
        model_value = arguments.model
        source_config_dir = None
        model_manifest_value = arguments.model_manifest
    else:
        policy_table_value = config.get("policy_table")
        model_value = config.get("model")
        source_config_dir = config_dir
        model_manifest_value = config.get("model_manifest")
    if (policy_table_value is None) == (model_value is None):
        raise CliUsageError(
            "exactly one of --policy-table or --model must be provided"
        )
    if policy_table_value is not None and model_manifest_value is not None:
        raise CliUsageError(
            "--model-manifest may only be used with --model"
        )

    if cli_source_supplied:
        acknowledge_trusted_model = arguments.acknowledge_trusted_model
        if acknowledge_trusted_model is None:
            acknowledge_trusted_model = False
    else:
        acknowledge_trusted_model = _configured(
            arguments,
            config,
            "acknowledge_trusted_model",
            default=False,
        )
    if type(acknowledge_trusted_model) is not bool:
        raise CliUsageError(
            "acknowledge_trusted_model must be a boolean in --config"
        )
    if model_value is None and acknowledge_trusted_model:
        raise CliUsageError(
            "--acknowledge-trusted-model may only be used with --model"
        )
    if model_value is not None and not acknowledge_trusted_model:
        raise CliUsageError(
            "--model requires --acknowledge-trusted-model to confirm that the "
            "archive came from a trusted source"
        )

    model_path = (
        None
        if model_value is None
        else _path_value(
            model_value,
            option="--model",
            config_dir=source_config_dir,
        )
    )
    if model_path is None:
        model_manifest_path = None
    elif model_manifest_value is None:
        from stache.explainability.model_manifest import manifest_path_for_model

        model_manifest_path = manifest_path_for_model(model_path)
    else:
        model_manifest_path = _path_value(
            model_manifest_value,
            option="--model-manifest",
            config_dir=source_config_dir,
        )

    minimum_basis = _exact_string_choice(
        _configured(
            arguments,
            config,
            "minimum_basis",
            default="graph_boundary",
        ),
        option="--minimum-basis",
        choices=("graph_boundary", "formal_global"),
    )
    counterfactuals = _exact_string_choice(
        _configured(
            arguments,
            config,
            "counterfactuals",
            default="both",
        ),
        option="--counterfactuals",
        choices=("minimal", "boundary", "both"),
    )
    extent = _exact_string_choice(
        _configured(arguments, config, "extent", default="exact"),
        option="--extent",
        choices=("exact", "through_minimal_cf"),
    )
    if extent == "through_minimal_cf" and counterfactuals != "minimal":
        raise CliUsageError(
            "--extent through_minimal_cf requires --counterfactuals minimal"
        )

    output_value = _configured(arguments, config, "output")
    if output_value is None:
        raise CliUsageError("--output is required")
    output_dir = None if arguments.output is not None else config_dir
    output = _path_value(output_value, option="--output", config_dir=output_dir)
    if not output.parent.is_dir():
        raise CliUsageError(f"--output parent directory does not exist: {output.parent}")

    overwrite = _configured(arguments, config, "overwrite", default=False)
    if type(overwrite) is not bool:
        raise CliUsageError("--overwrite must be a boolean in --config")
    if output.exists() and not overwrite:
        raise CliUsageError(
            f"--output already exists; pass --overwrite to replace it: {output}"
        )

    return {
        "domain": domain,
        "state_universe": state_universe,
        "seed": seed,
        "policy_table": (
            None
            if policy_table_value is None
            else _path_value(
                policy_table_value,
                option="--policy-table",
                config_dir=source_config_dir,
            )
        ),
        "model": model_path,
        "model_manifest": model_manifest_path,
        "acknowledge_trusted_model": acknowledge_trusted_model,
        "minimum_basis": minimum_basis,
        "counterfactuals": counterfactuals,
        "extent": extent,
        "max_expanded": _exact_integer(
            _configured(arguments, config, "max_expanded"),
            option="--max-expanded",
            minimum=0,
        ),
        "max_policy_queries": _exact_integer(
            _configured(arguments, config, "max_policy_queries"),
            option="--max-policy-queries",
            minimum=1,
        ),
        "max_graph_depth": _exact_integer(
            _configured(arguments, config, "max_graph_depth"),
            option="--max-graph-depth",
            minimum=0,
        ),
        "output": output,
        "overwrite": overwrite,
    }


def _load_policy_table(path: Path) -> dict[int, object]:
    raw = _safe_mapping(
        path,
        label="--policy-table",
        require_string_keys=False,
    )
    table: dict[int, object] = {}
    for raw_key, action in raw.items():
        if type(raw_key) is int:
            key = raw_key
        elif isinstance(raw_key, str) and raw_key.isascii() and raw_key.isdecimal():
            key = int(raw_key)
        else:
            raise CliUsageError(
                "--policy-table keys must be Taxi indices encoded as integers"
            )
        if not 0 <= key <= 499:
            raise CliUsageError(f"--policy-table key is outside [0, 499]: {key}")
        if key in table:
            raise CliUsageError(f"--policy-table contains duplicate Taxi key {key}")
        table[key] = action
    return table


def _snapshot_model(path: Path) -> tuple[BytesIO, str]:
    """Read once so the fingerprint and SB3 loader consume identical bytes."""

    if not path.is_file():
        raise CliUsageError(f"model file does not exist or is not regular: {path}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CliUsageError(f"cannot read model file {path}: {error}") from error
    fingerprint = f"sha256:{sha256(payload).hexdigest()}"
    return BytesIO(payload), fingerprint


def _provenance() -> dict[str, object]:
    dependencies: dict[str, str] = {"python": platform.python_version()}
    for package in (
        "stache",
        "numpy",
        "stable-baselines3",
        "torch",
        "pyyaml",
    ):
        try:
            dependencies[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    result: dict[str, object] = {"dependencies": dependencies}
    repository = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / ".git").exists()
        ),
        None,
    )
    if repository is None:
        return result
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return result
    if revision.returncode == 0 and status.returncode == 0:
        result["git"] = {
            "commit": revision.stdout.strip(),
            "dirty": bool(status.stdout),
        }
    return result


def _run_compute_rr(config: Mapping[str, Any]) -> int:
    if config["policy_table"] is None and config.get(
        "acknowledge_trusted_model"
    ) is not True:
        raise CliUsageError(
            "--model requires --acknowledge-trusted-model to confirm that the "
            "archive came from a trusted source"
        )

    from stache.explainability.artifacts import ArtifactError, save_result
    from stache.explainability.connectors import TaxiConnector
    from stache.explainability.model_manifest import (
        ModelManifestError,
        load_model_manifest,
    )
    from stache.explainability.core import (
        CounterfactualSelection,
        MinimumBasis,
        ModelActionOracle,
        SearchExtent,
        SearchOptions,
        TableActionOracle,
        compute_rr,
    )

    connector = TaxiConnector()
    if connector.identity.state_universe != config["state_universe"]:
        raise CliUsageError(
            "selected --state-universe does not match the Taxi connector identity"
        )
    seed = connector.decode_index(config["seed"])
    policy_table = config["policy_table"]
    if policy_table is not None:
        oracle = TableActionOracle(connector, _load_policy_table(policy_table))
    else:
        model_path = config["model"]
        model_snapshot, fingerprint = _snapshot_model(model_path)
        try:
            manifest = load_model_manifest(config["model_manifest"])
        except ModelManifestError as error:
            raise CliUsageError(str(error)) from error
        from stable_baselines3 import DQN

        try:
            model = DQN.load(model_snapshot, env=None)
        except Exception as error:
            raise CliExecutionError(
                f"could not load DQN model {model_path}: {error}"
            ) from error
        oracle = ModelActionOracle(
            connector,
            model,
            source_fingerprint=fingerprint,
            manifest=manifest,
        )

    options = SearchOptions(
        minimum_basis=MinimumBasis(config["minimum_basis"]),
        counterfactuals=CounterfactualSelection(config["counterfactuals"]),
        extent=SearchExtent(config["extent"]),
        max_expanded=config["max_expanded"],
        max_policy_queries=config["max_policy_queries"],
        max_graph_depth=config["max_graph_depth"],
    )
    result = compute_rr(seed, connector, oracle, options)
    try:
        save_result(
            config["output"],
            result,
            connector,
            provenance=_provenance(),
            overwrite=config["overwrite"],
        )
    except ArtifactError as error:
        raise CliExecutionError(f"could not save RR artifact: {error}") from error
    print(
        json.dumps(
            {
                "output": str(config["output"]),
                "stop_reason": result.completeness.stop_reason.value,
                "region_size": len(result.region),
                "boundary_counterfactuals": len(
                    result.boundary_counterfactuals
                ),
                "minimal_counterfactuals": len(result.minimal_counterfactuals),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch the installed ``stache`` command."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return 0
    if arguments.command != "compute-rr":  # pragma: no cover - parser invariant
        parser.error(f"unknown command: {arguments.command}")

    try:
        config = _validated_compute_config(arguments)
        return _run_compute_rr(config)
    except CliUsageError as error:
        parser.error(str(error))
    except CliExecutionError as error:
        print(f"stache compute-rr: error: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError) as error:
        print(f"stache compute-rr: error: {error}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
