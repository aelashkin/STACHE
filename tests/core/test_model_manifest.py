"""Model manifest persistence is bound to the exact model bytes."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from stache.explainability.connectors.taxi import TaxiConnector
from stache.explainability.model_manifest import (
    ModelManifestError,
    load_model_manifest,
    write_connector_model_manifest,
)


def test_write_connector_model_manifest_binds_exact_model_and_connector(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.zip"
    model_bytes = b"deterministic-model-archive"
    model_path.write_bytes(model_bytes)
    connector = TaxiConnector()

    manifest_path = write_connector_model_manifest(model_path, connector)

    assert manifest_path == tmp_path / "model.manifest.yaml"
    manifest = load_model_manifest(manifest_path)
    assert manifest.model_fingerprint == f"sha256:{sha256(model_bytes).hexdigest()}"
    assert manifest.observation_identity == connector.observation_spec.identity
    assert manifest.action_spec == connector.action_spec


def test_write_connector_model_manifest_requires_explicit_overwrite(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"first-model")
    connector = TaxiConnector()
    manifest_path = write_connector_model_manifest(model_path, connector)

    model_path.write_bytes(b"second-model")
    with pytest.raises(FileExistsError, match="already exists"):
        write_connector_model_manifest(model_path, connector)

    written_path = write_connector_model_manifest(
        model_path,
        connector,
        overwrite=True,
    )

    assert written_path == manifest_path
    assert load_model_manifest(manifest_path).model_fingerprint == (
        f"sha256:{sha256(b'second-model').hexdigest()}"
    )
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_connector_model_manifest_rejects_model_symlink(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "target.zip"
    target_path.write_bytes(b"model")
    model_path = tmp_path / "model.zip"
    model_path.symlink_to(target_path)

    with pytest.raises(ModelManifestError, match="regular non-symlink"):
        write_connector_model_manifest(model_path, TaxiConnector())


def test_load_model_manifest_rejects_symlink_and_oversized_inputs(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"model")
    real_manifest = write_connector_model_manifest(model_path, TaxiConnector())
    linked_manifest = tmp_path / "linked.manifest.yaml"
    linked_manifest.symlink_to(real_manifest)

    with pytest.raises(ModelManifestError, match="regular non-symlink"):
        load_model_manifest(linked_manifest)

    oversized_manifest = tmp_path / "oversized.manifest.yaml"
    oversized_manifest.write_bytes(b"x" * (64 * 1024 + 1))
    with pytest.raises(ModelManifestError, match="exceeds"):
        load_model_manifest(oversized_manifest)
