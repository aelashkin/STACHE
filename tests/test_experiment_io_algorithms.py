"""Compatibility coverage for experiment model algorithm dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from stache.explainability.connectors.taxi import TaxiConnector
from stache.explainability.model_manifest import load_model_manifest
from stache.utils import experiment_io


def test_load_experiment_requires_trust_before_any_file_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        experiment_io.UntrustedModelError,
        match="trusted source",
    ):
        experiment_io.load_experiment(str(tmp_path))


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
    loaded_payloads: list[bytes] = []

    class FakeDQN:
        @staticmethod
        def load(source: object) -> object:
            loaded_payloads.append(source.getvalue())
            return sentinel

    monkeypatch.setattr(experiment_io, "DQN", FakeDQN)

    model, config = experiment_io.load_experiment(
        str(tmp_path),
        acknowledge_trusted_model=True,
    )

    assert model is sentinel
    assert loaded_payloads == [b""]
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

    _, config = experiment_io.load_experiment(
        str(tmp_path),
        acknowledge_trusted_model=True,
    )

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
        experiment_io.load_experiment(
            str(tmp_path),
            acknowledge_trusted_model=True,
        )


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


def test_load_config_rejects_oversized_and_symlink_inputs(
    tmp_path: Path,
) -> None:
    oversized = tmp_path / "oversized.yaml"
    with oversized.open("wb") as stream:
        stream.truncate(experiment_io.EXPERIMENT_CONFIG_MAX_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds"):
        experiment_io.load_config(oversized)

    real = tmp_path / "real.yaml"
    real.write_text("model_type: DQN\n", encoding="utf-8")
    linked = tmp_path / "linked.yaml"
    linked.symlink_to(real)
    with pytest.raises(ValueError, match="non-symlink"):
        experiment_io.load_config(linked)


def test_save_experiment_can_bind_saved_model_to_explicit_connector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_bytes = b"saved-model-archive"

    class FakeModel:
        def save(self, path: str) -> None:
            Path(path).write_bytes(model_bytes)

    connector = TaxiConnector()
    saved = experiment_io.save_experiment(
        FakeModel(),
        {"env_name": "Taxi-v3"},
        {"model_type": "DQN"},
        "training complete",
        experiment_dir=str(tmp_path),
        model_connector=connector,
    )

    manifest_path = Path(saved["manifest_path"])
    manifest = load_model_manifest(manifest_path)
    assert manifest_path == tmp_path / "model.manifest.yaml"
    assert manifest.observation_identity == connector.observation_spec.identity
    assert manifest.action_spec == connector.action_spec
    assert set(path.name for path in tmp_path.iterdir()) == {
        "config.yaml",
        "model.manifest.yaml",
        "model.zip",
        "training.log",
    }

    class LoadedModel:
        pass

    class FakeDQN:
        @staticmethod
        def load(source: object) -> LoadedModel:
            assert source.getvalue() == model_bytes
            return LoadedModel()

    monkeypatch.setattr(experiment_io, "DQN", FakeDQN)
    loaded, _ = experiment_io.load_experiment(
        str(tmp_path),
        acknowledge_trusted_model=True,
        model_connector=connector,
    )
    assert loaded.stache_model_manifest == manifest
