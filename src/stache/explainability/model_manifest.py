"""Safe persistence for model-owned semantic manifests."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import stat
import tempfile

import yaml

from stache.explainability.core.policy import (
    ModelManifest,
    PolicyConfigurationError,
    model_manifest_from_document,
    model_manifest_to_document,
)
from stache.explainability.core.connector import (
    DiscreteActionSpec,
    ObservationIdentity,
)
from stache.utils.safe_yaml import safe_load_unique


MODEL_MANIFEST_FILENAME = "model.manifest.yaml"
MODEL_MANIFEST_MAX_BYTES = 64 * 1024


class ModelManifestError(ValueError):
    """A model manifest is missing, malformed, or semantically invalid."""


def manifest_path_for_model(model_path: Path) -> Path:
    """Return the conventional sidecar path for a model archive."""

    return model_path.with_name(MODEL_MANIFEST_FILENAME)


def load_model_manifest(path: Path) -> ModelManifest:
    """Load a strict primitive-only manifest with duplicate-key rejection."""

    manifest_path = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(manifest_path, flags)
    except OSError as error:
        raise ModelManifestError(
            "model manifest must be a readable regular non-symlink file: "
            f"{manifest_path}"
        ) from error
    try:
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise ModelManifestError(
                "model manifest must be a readable regular non-symlink file: "
                f"{manifest_path}"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read(MODEL_MANIFEST_MAX_BYTES + 1)
        if len(payload) > MODEL_MANIFEST_MAX_BYTES:
            raise ModelManifestError(
                f"model manifest exceeds {MODEL_MANIFEST_MAX_BYTES} bytes: "
                f"{manifest_path}"
            )
        document = safe_load_unique(payload.decode("utf-8"))
    except ModelManifestError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ModelManifestError(
            f"cannot safely load model manifest {manifest_path}: {error}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        return model_manifest_from_document(document)
    except PolicyConfigurationError as error:
        raise ModelManifestError(
            f"invalid model manifest {manifest_path}: {error}"
        ) from error


def model_file_fingerprint(path: Path) -> str:
    """Return a SHA-256 identity for one regular, non-symlink model file."""

    model_path = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(model_path, flags)
    except OSError as error:
        raise ModelManifestError(
            f"model archive does not exist or is not a readable regular "
            f"non-symlink file: "
            f"{model_path}"
        ) from error

    digest = sha256()
    try:
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise ModelManifestError(
                "model archive does not exist or is not a readable regular "
                "non-symlink file: "
                f"{model_path}"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ModelManifestError(
            f"cannot fingerprint model archive {model_path}: {error}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return f"sha256:{digest.hexdigest()}"


def snapshot_model_file(path: Path) -> tuple[BytesIO, str]:
    """Read one regular model file once for identical hashing and loading."""

    model_path = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(model_path, flags)
    except OSError as error:
        raise ModelManifestError(
            f"model archive does not exist or is not a readable regular "
            f"non-symlink file: "
            f"{model_path}"
        ) from error

    try:
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise ModelManifestError(
                "model archive does not exist or is not a readable regular "
                "non-symlink file: "
                f"{model_path}"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read()
    except OSError as error:
        raise ModelManifestError(
            f"cannot snapshot model archive {model_path}: {error}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    fingerprint = f"sha256:{sha256(payload).hexdigest()}"
    return BytesIO(payload), fingerprint


def write_connector_model_manifest(
    model_path: Path,
    connector: object,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically persist model bytes and connector semantics as one sidecar."""

    try:
        observation_identity = connector.observation_spec.identity
        action_spec = connector.action_spec
    except AttributeError as error:
        raise ModelManifestError(
            "connector must declare observation_spec.identity and action_spec"
        ) from error
    if not isinstance(observation_identity, ObservationIdentity):
        raise ModelManifestError(
            "connector observation identity must be an ObservationIdentity"
        )
    if not isinstance(action_spec, DiscreteActionSpec):
        raise ModelManifestError(
            "connector action contract must be a DiscreteActionSpec"
        )

    model_path = Path(model_path)
    manifest_path = manifest_path_for_model(model_path)
    if manifest_path.is_symlink():
        raise ModelManifestError(
            f"refusing to replace symlink model manifest: {manifest_path}"
        )
    manifest = ModelManifest(
        model_fingerprint=model_file_fingerprint(model_path),
        observation_identity=observation_identity,
        action_spec=action_spec,
    )
    serialized = yaml.safe_dump(
        model_manifest_to_document(manifest),
        sort_keys=False,
        allow_unicode=True,
    )

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())

        if overwrite:
            if manifest_path.is_symlink():
                raise ModelManifestError(
                    f"refusing to replace symlink model manifest: {manifest_path}"
                )
            os.replace(temporary_path, manifest_path)
            temporary_path = None
        else:
            try:
                os.link(
                    temporary_path,
                    manifest_path,
                    follow_symlinks=False,
                )
            except FileExistsError as error:
                raise FileExistsError(
                    f"model manifest already exists: {manifest_path}; pass "
                    "overwrite=True to replace it"
                ) from error
        return manifest_path
    except (ModelManifestError, FileExistsError):
        raise
    except OSError as error:
        raise ModelManifestError(
            f"cannot safely write model manifest {manifest_path}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "MODEL_MANIFEST_FILENAME",
    "MODEL_MANIFEST_MAX_BYTES",
    "ModelManifestError",
    "load_model_manifest",
    "manifest_path_for_model",
    "model_file_fingerprint",
    "snapshot_model_file",
    "write_connector_model_manifest",
]
