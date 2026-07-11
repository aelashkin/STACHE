"""Safe persistence for model-owned semantic manifests."""

from __future__ import annotations

from pathlib import Path

import yaml

from stache.explainability.core.policy import (
    ModelManifest,
    PolicyConfigurationError,
    model_manifest_from_document,
)
from stache.utils.safe_yaml import safe_load_unique


MODEL_MANIFEST_FILENAME = "model.manifest.yaml"


class ModelManifestError(ValueError):
    """A model manifest is missing, malformed, or semantically invalid."""


def manifest_path_for_model(model_path: Path) -> Path:
    """Return the conventional sidecar path for a model archive."""

    return model_path.with_name(MODEL_MANIFEST_FILENAME)


def load_model_manifest(path: Path) -> ModelManifest:
    """Load a strict primitive-only manifest with duplicate-key rejection."""

    if not path.is_file():
        raise ModelManifestError(
            f"model manifest does not exist or is not regular: {path}"
        )
    try:
        document = safe_load_unique(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ModelManifestError(
            f"cannot safely load model manifest {path}: {error}"
        ) from error
    try:
        return model_manifest_from_document(document)
    except PolicyConfigurationError as error:
        raise ModelManifestError(
            f"invalid model manifest {path}: {error}"
        ) from error


__all__ = [
    "MODEL_MANIFEST_FILENAME",
    "ModelManifestError",
    "load_model_manifest",
    "manifest_path_for_model",
]
