"""All Taxi model-loading commands share one fail-closed boundary."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from stache.explainability.connectors.taxi import TaxiConnector
from stache.explainability.core.policy import ModelCompatibilityError
from stache.explainability.model_manifest import write_connector_model_manifest
from stache.explainability.taxi.model_loading import (
    TrustedModelRequiredError,
    load_trusted_taxi_model,
)


def test_trust_acknowledgement_is_required_before_model_file_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(TrustedModelRequiredError, match="trusted source"):
        load_trusted_taxi_model(
            tmp_path / "missing.zip",
            acknowledge_trusted_model=False,
        )


def test_loader_hashes_and_deserializes_one_immutable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b"original-model-snapshot"
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(original)
    write_connector_model_manifest(model_path, TaxiConnector())
    sentinel = object()
    captured: list[bytes] = []

    def load_snapshot(
        source: object,
        *,
        env: object = None,
        print_system_info: bool = False,
    ) -> object:
        assert env is None
        assert print_system_info is False
        assert isinstance(source, BytesIO)
        captured.append(source.getvalue())
        model_path.write_bytes(b"changed-after-snapshot")
        return sentinel

    from stable_baselines3 import DQN

    monkeypatch.setattr(DQN, "load", staticmethod(load_snapshot))

    loaded = load_trusted_taxi_model(
        model_path,
        acknowledge_trusted_model=True,
    )

    assert loaded.model is sentinel
    assert loaded.model_fingerprint == loaded.manifest.model_fingerprint
    assert captured == [original]


def test_manifest_mismatch_is_rejected_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"first-model")
    write_connector_model_manifest(model_path, TaxiConnector())
    model_path.write_bytes(b"different-model")
    called = False

    def unexpected_load(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        return object()

    from stable_baselines3 import DQN

    monkeypatch.setattr(DQN, "load", staticmethod(unexpected_load))

    with pytest.raises(ModelCompatibilityError, match="fingerprint"):
        load_trusted_taxi_model(
            model_path,
            acknowledge_trusted_model=True,
        )

    assert called is False

