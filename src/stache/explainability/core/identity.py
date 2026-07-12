"""Canonical identity hashing shared by policy, search, and artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Mapping


class IdentityEncodingError(ValueError):
    """Identity material is not canonical primitive JSON data."""


def fingerprint_document(document: Mapping[str, object]) -> str:
    """Return a deterministic SHA-256 identity for primitive mapping data."""

    if not isinstance(document, Mapping) or any(
        type(key) is not str for key in document
    ):
        raise IdentityEncodingError(
            "identity document must be a string-keyed mapping"
        )
    try:
        payload = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise IdentityEncodingError(
            f"identity document is not canonical primitive JSON: {error}"
        ) from error
    return f"sha256:{sha256(payload).hexdigest()}"


__all__ = ["IdentityEncodingError", "fingerprint_document"]
