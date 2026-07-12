"""Shared trust and semantic boundary for Taxi model deserialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stache.explainability.connectors.taxi import TaxiConnector
from stache.explainability.core.policy import (
    ModelManifest,
    validate_model_manifest_binding,
)
from stache.explainability.model_manifest import (
    load_model_manifest,
    manifest_path_for_model,
    snapshot_model_file,
)


class TrustedModelRequiredError(ValueError):
    """Model deserialization was requested without an explicit trust decision."""


@dataclass(frozen=True, slots=True)
class LoadedTaxiModel:
    """One model loaded from the same bytes used for its semantic identity."""

    model: object
    model_fingerprint: str
    manifest: ModelManifest


def load_trusted_taxi_model(
    model_path: Path,
    *,
    acknowledge_trusted_model: bool,
) -> LoadedTaxiModel:
    """Validate trust, bytes, and Taxi semantics before loading a DQN."""

    if acknowledge_trusted_model is not True:
        raise TrustedModelRequiredError(
            "Taxi model loading requires explicit acknowledgement that the "
            "archive came from a trusted source"
        )

    model_path = Path(model_path)
    connector = TaxiConnector()
    manifest = load_model_manifest(manifest_path_for_model(model_path))
    model_snapshot, model_fingerprint = snapshot_model_file(model_path)
    validate_model_manifest_binding(
        connector,
        model_fingerprint,
        manifest,
    )

    from stable_baselines3 import DQN

    model = DQN.load(
        model_snapshot,
        env=None,
        print_system_info=False,
    )
    return LoadedTaxiModel(
        model=model,
        model_fingerprint=model_fingerprint,
        manifest=manifest,
    )


__all__ = [
    "LoadedTaxiModel",
    "TrustedModelRequiredError",
    "load_trusted_taxi_model",
]
