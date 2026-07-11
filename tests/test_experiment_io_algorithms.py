"""Compatibility coverage for experiment model algorithm dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from stache.utils import experiment_io


def test_load_experiment_supports_dqn_without_constructing_an_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.zip"
    model_path.touch()
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "env_config": {"env_name": "Taxi-v3"},
                "model_config": {"model_type": "DQN"},
            }
        ),
        encoding="utf-8",
    )
    sentinel = object()
    loaded_paths: list[str] = []

    class FakeDQN:
        @staticmethod
        def load(path: str) -> object:
            loaded_paths.append(path)
            return sentinel

    monkeypatch.setattr(experiment_io, "DQN", FakeDQN)

    model, config = experiment_io.load_experiment(str(tmp_path))

    assert model is sentinel
    assert loaded_paths == [str(model_path)]
    assert config["model_config"]["model_type"] == "DQN"


def test_load_experiment_accepts_only_the_historical_python_tuple_tag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.zip"
    model_path.touch()
    (tmp_path / "config.yaml").write_text(
        "env_config:\n"
        "  env_name: Taxi-v3\n"
        "model_config:\n"
        "  model_type: DQN\n"
        "  train_freq: !!python/tuple\n"
        "  - 1\n"
        "  - step\n",
        encoding="utf-8",
    )

    class FakeDQN:
        @staticmethod
        def load(path: str) -> object:
            return object()

    monkeypatch.setattr(experiment_io, "DQN", FakeDQN)

    _, config = experiment_io.load_experiment(str(tmp_path))

    assert config["model_config"]["train_freq"] == (1, "step")


def test_load_experiment_still_rejects_other_python_specific_tags(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.zip").touch()
    (tmp_path / "config.yaml").write_text(
        "env_config: {}\n"
        "model_config:\n"
        "  model_type: DQN\n"
        "  dangerous: !!python/object/apply:builtins.str [unsafe]\n",
        encoding="utf-8",
    )

    with pytest.raises(yaml.constructor.ConstructorError):
        experiment_io.load_experiment(str(tmp_path))


def test_save_config_emits_safe_primitive_yaml_for_tuple_values(
    tmp_path: Path,
) -> None:
    path = experiment_io.save_config(
        {"env_name": "Taxi-v3"},
        {"model_type": "DQN", "train_freq": (1, "step")},
        str(tmp_path),
    )

    serialized = Path(path).read_text(encoding="utf-8")
    assert "!!python" not in serialized
    assert yaml.safe_load(serialized)["model_config"]["train_freq"] == [1, "step"]
