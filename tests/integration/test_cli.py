"""Installed-command smoke contracts for the composable RR CLI."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from importlib import metadata
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from stache import cli


def installed_stache() -> str:
    entry_points = [
        entry_point
        for entry_point in metadata.entry_points(group="console_scripts")
        if entry_point.name == "stache"
    ]
    assert len(entry_points) == 1, "the wheel must install exactly one `stache` command"
    executable = shutil.which("stache")
    assert executable is not None, "the installed `stache` executable is not on PATH"
    return executable


def run_stache(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [installed_stache(), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def write_constant_taxi_policy(path: Path, *, action: int = 0) -> Path:
    path.write_text(
        yaml.safe_dump({state: action for state in range(500)}),
        encoding="utf-8",
    )
    return path


def successful_arguments(
    tmp_path: Path,
    *,
    seed: int = 0,
    output_name: str = "result.yaml",
) -> list[str]:
    policy = write_constant_taxi_policy(tmp_path / "constant-policy.yaml")
    return [
        "compute-rr",
        "--domain",
        "taxi",
        "--state-universe",
        "taxi-factored-500",
        "--seed",
        str(seed),
        "--policy-table",
        str(policy),
        "--minimum-basis",
        "graph_boundary",
        "--counterfactuals",
        "both",
        "--extent",
        "exact",
        "--output",
        str(tmp_path / output_name),
    ]


def test_installed_root_help_lists_compute_rr() -> None:
    completed = run_stache("--help")

    assert completed.returncode == 0, completed.stderr
    assert "compute-rr" in completed.stdout


def test_installed_compute_rr_help_exposes_scientific_and_budget_options() -> None:
    completed = run_stache("compute-rr", "--help")

    assert completed.returncode == 0, completed.stderr
    for option in (
        "--domain",
        "--state-universe",
        "--seed",
        "--policy-table",
        "--model",
        "--minimum-basis",
        "--counterfactuals",
        "--extent",
        "--max-expanded",
        "--max-policy-queries",
        "--max-graph-depth",
        "--config",
        "--output",
        "--overwrite",
    ):
        assert option in completed.stdout


def complete_arguments(tmp_path: Path) -> list[str]:
    policy = tmp_path / "policy.yaml"
    policy.write_text("{}\n", encoding="utf-8")
    return [
        "compute-rr",
        "--domain",
        "taxi",
        "--state-universe",
        "taxi-factored-500",
        "--seed",
        "0",
        "--policy-table",
        str(policy),
        "--minimum-basis",
        "graph_boundary",
        "--counterfactuals",
        "both",
        "--extent",
        "exact",
        "--max-expanded",
        "100",
        "--output",
        str(tmp_path / "result.yaml"),
    ]


def test_invalid_domain_is_rejected_before_any_policy_load(tmp_path: Path) -> None:
    arguments = complete_arguments(tmp_path)
    arguments[arguments.index("taxi")] = "minigrid"

    completed = run_stache(*arguments)

    assert completed.returncode != 0
    assert "domain" in completed.stderr.lower()
    assert "taxi" in completed.stderr.lower()
    assert not (tmp_path / "result.yaml").exists()


def test_invalid_state_universe_is_rejected_before_policy_load(
    tmp_path: Path,
) -> None:
    arguments = complete_arguments(tmp_path)
    arguments[arguments.index("taxi-factored-500")] = "taxi-reachable-404"

    completed = run_stache(*arguments)

    assert completed.returncode != 0
    assert "state-universe" in completed.stderr.lower()
    assert "taxi-factored-500" in completed.stderr.lower()
    assert not (tmp_path / "result.yaml").exists()


def test_config_state_universe_is_validated_before_relative_policy_load(
    tmp_path: Path,
) -> None:
    config = tmp_path / "compute.yaml"
    config.write_text(
        "domain: taxi\n"
        "state_universe: taxi-reachable-404\n"
        "seed: 0\n"
        "policy_table: missing-policy.yaml\n"
        "output: result.yaml\n",
        encoding="utf-8",
    )

    completed = run_stache("compute-rr", "--config", str(config))

    assert completed.returncode != 0
    assert "state-universe" in completed.stderr.lower()
    assert "taxi-factored-500" in completed.stderr.lower()
    assert not (tmp_path / "result.yaml").exists()


def test_safe_config_resolves_policy_and_output_relative_to_its_directory(
    tmp_path: Path,
) -> None:
    write_constant_taxi_policy(tmp_path / "policy.yaml")
    config = tmp_path / "compute.yaml"
    config.write_text(
        "domain: taxi\n"
        "state_universe: taxi-factored-500\n"
        "seed: 0\n"
        "policy_table: policy.yaml\n"
        "minimum_basis: graph_boundary\n"
        "counterfactuals: both\n"
        "extent: exact\n"
        "max_expanded: 0\n"
        "output: result.yaml\n",
        encoding="utf-8",
    )

    completed = run_stache("compute-rr", "--config", str(config))

    assert completed.returncode == 0, completed.stderr
    document = yaml.safe_load((tmp_path / "result.yaml").read_text(encoding="utf-8"))
    assert document["options"]["max_expanded"] == 0
    assert document["result"]["completeness"]["stop_reason"] == "max_expanded"


def test_config_rejects_duplicate_nested_keys_before_writing(
    tmp_path: Path,
) -> None:
    write_constant_taxi_policy(tmp_path / "policy.yaml")
    config = tmp_path / "compute.yaml"
    config.write_text(
        "compute_rr:\n"
        "  domain: taxi\n"
        "  state_universe: taxi-factored-500\n"
        "  seed: 0\n"
        "  seed: 1\n"
        "  policy_table: policy.yaml\n"
        "  max_expanded: 0\n"
        "  output: result.yaml\n",
        encoding="utf-8",
    )

    completed = run_stache("compute-rr", "--config", str(config))

    assert completed.returncode != 0
    assert "duplicate" in completed.stderr.lower()
    assert not (tmp_path / "result.yaml").exists()


@pytest.mark.parametrize(
    "serialized",
    [
        pytest.param("0: 0\n0: 1\n", id="exact-yaml-key"),
        pytest.param('0: 0\n"0": 1\n', id="normalized-taxi-key"),
    ],
)
def test_policy_table_rejects_duplicate_keys_before_writing(
    tmp_path: Path,
    serialized: str,
) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text(serialized, encoding="utf-8")
    target = tmp_path / "result.yaml"

    completed = run_stache(
        "compute-rr",
        "--domain",
        "taxi",
        "--state-universe",
        "taxi-factored-500",
        "--seed",
        "0",
        "--policy-table",
        str(policy),
        "--max-expanded",
        "0",
        "--output",
        str(target),
    )

    assert completed.returncode != 0
    assert "duplicate" in completed.stderr.lower()
    assert not target.exists()


@pytest.mark.parametrize(
    "option, invalid_value",
    [
        pytest.param("--minimum-basis", "nearest-ish", id="minimum-basis"),
        pytest.param("--counterfactuals", "first-only", id="selection"),
        pytest.param("--extent", "forever", id="extent"),
        pytest.param("--max-expanded", "-1", id="negative-budget"),
    ],
)
def test_invalid_option_values_fail_without_writing_artifacts(
    tmp_path: Path,
    option: str,
    invalid_value: str,
) -> None:
    arguments = complete_arguments(tmp_path)
    index = arguments.index(option)
    arguments[index + 1] = invalid_value

    completed = run_stache(*arguments)

    assert completed.returncode != 0
    assert option in completed.stderr
    assert not (tmp_path / "result.yaml").exists()


def test_policy_table_and_model_are_mutually_exclusive(tmp_path: Path) -> None:
    arguments = complete_arguments(tmp_path)
    arguments.extend(["--model", str(tmp_path / "model.zip")])

    completed = run_stache(*arguments)

    assert completed.returncode != 0
    assert "--policy-table" in completed.stderr
    assert "--model" in completed.stderr
    assert not (tmp_path / "result.yaml").exists()


def test_complete_constant_policy_writes_safe_current_schema_artifact(
    tmp_path: Path,
) -> None:
    completed = run_stache(*successful_arguments(tmp_path))

    assert completed.returncode == 0, completed.stderr
    target = tmp_path / "result.yaml"
    serialized = target.read_text(encoding="utf-8")
    document = yaml.safe_load(serialized)
    assert "!!python" not in serialized
    assert document["schema"] == "stache.rr-result"
    assert document["schema_version"] == 1
    assert document["connector"]["state_universe"] == "taxi-factored-500"
    assert len(document["result"]["region"]) == 500
    assert document["result"]["completeness"]["region_complete"] is True
    assert document["result"]["completeness"]["boundary_complete"] is True
    assert document["result"]["stop_reason"] == "completed"
    assert document["result"]["continuation"] is None
    assert document["provenance"]["dependencies"]["python"].startswith("3.11.")
    if "git" in document["provenance"]:
        assert document["provenance"]["git"]["commit"]
        assert type(document["provenance"]["git"]["dirty"]) is bool


def test_zero_expansion_writes_truthful_nonresumable_frontier_summary(
    tmp_path: Path,
) -> None:
    arguments = successful_arguments(tmp_path)
    arguments[arguments.index("--output"):arguments.index("--output")] = [
        "--max-expanded",
        "0",
    ]

    completed = run_stache(*arguments)

    assert completed.returncode == 0, completed.stderr
    document = yaml.safe_load((tmp_path / "result.yaml").read_text(encoding="utf-8"))
    completeness = document["result"]["completeness"]
    continuation = document["result"]["continuation"]
    assert completeness["region_complete"] is False
    assert completeness["boundary_complete"] is False
    assert completeness["stop_reason"] == "max_expanded"
    assert completeness["remaining_frontier_size"] == 1
    assert continuation["resumable"] is False
    assert continuation["remaining_frontier_size"] == 1
    assert continuation["checkpoint_version"]
    assert continuation["payload_digest"].startswith("sha256:")


def test_overwrite_replaces_an_existing_result_artifact(tmp_path: Path) -> None:
    first = run_stache(*successful_arguments(tmp_path, seed=0))
    assert first.returncode == 0, first.stderr
    target = tmp_path / "result.yaml"
    first_document = yaml.safe_load(target.read_text(encoding="utf-8"))

    replacement_arguments = successful_arguments(tmp_path, seed=1)
    replacement_arguments.append("--overwrite")
    replacement = run_stache(*replacement_arguments)

    assert replacement.returncode == 0, replacement.stderr
    replacement_document = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert replacement_document["result"]["seed"]["state"] == [0, 0, 0, 1]
    assert replacement_document != first_document


def test_committed_dqn_model_writes_a_fingerprinted_budget_result(
    tmp_path: Path,
) -> None:
    model = (
        Path(__file__).parents[2]
        / "data"
        / "experiments"
        / "models"
        / "Taxi-v3_DQN_model_100"
        / "model.zip"
    )
    target = tmp_path / "dqn-result.yaml"

    completed = run_stache(
        "compute-rr",
        "--domain",
        "taxi",
        "--state-universe",
        "taxi-factored-500",
        "--seed",
        "0",
        "--model",
        str(model),
        "--minimum-basis",
        "graph_boundary",
        "--counterfactuals",
        "both",
        "--extent",
        "exact",
        "--max-expanded",
        "0",
        "--output",
        str(target),
    )

    assert completed.returncode == 0, completed.stderr
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    fingerprint = document["policy"]["fingerprint"]
    assert document["policy"]["source"]["source"] == "model"
    assert fingerprint.startswith("sha256:")
    assert document["policy"]["source"]["fingerprint"] == fingerprint
    assert document["result"]["stats"]["model_queries"] == 1
    assert document["result"]["completeness"]["stop_reason"] == "max_expanded"


def test_model_fingerprint_and_load_use_the_same_immutable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b"trusted model snapshot"
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(original)
    target = tmp_path / "result.yaml"
    captured: list[bytes] = []

    class SnapshotModel:
        observation_space = SimpleNamespace(
            shape=(500,),
            dtype=np.dtype("float32"),
        )
        action_space = SimpleNamespace(n=6)

        def predict(
            self,
            observation: np.ndarray,
            *,
            deterministic: bool = False,
        ) -> tuple[np.ndarray, None]:
            assert observation.shape == (500,)
            assert deterministic is True
            return np.array([0], dtype=np.int64), None

    def load_snapshot(source: object, *, env: object = None) -> SnapshotModel:
        assert env is None
        assert isinstance(source, BytesIO)
        captured.append(source.getvalue())
        model_path.write_bytes(b"mutated after snapshot")
        return SnapshotModel()

    from stable_baselines3 import DQN

    monkeypatch.setattr(DQN, "load", staticmethod(load_snapshot))
    monkeypatch.setattr(cli, "_provenance", lambda: {})

    exit_code = cli._run_compute_rr(
        {
            "domain": "taxi",
            "state_universe": "taxi-factored-500",
            "seed": 0,
            "policy_table": None,
            "model": model_path,
            "minimum_basis": "graph_boundary",
            "counterfactuals": "both",
            "extent": "exact",
            "max_expanded": 0,
            "max_policy_queries": None,
            "max_graph_depth": None,
            "output": target,
            "overwrite": False,
        }
    )

    assert exit_code == 0
    assert captured == [original]
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert document["policy"]["fingerprint"] == (
        "sha256:" + sha256(original).hexdigest()
    )
