"""Strict SafeLoader helpers shared by untrusted YAML input boundaries."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any

import yaml


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping level."""


class SafeInputError(ValueError):
    """An untrusted text input violates the bounded regular-file contract."""


def read_bounded_regular_text(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> str:
    """Read bounded UTF-8 once without following a final-component symlink."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    target = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags)
    except OSError as error:
        raise SafeInputError(
            f"{label} does not exist or is not a readable regular non-symlink "
            f"file: {target}"
        ) from error
    try:
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise SafeInputError(
                f"{label} is not a readable regular non-symlink file: {target}"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise SafeInputError(
                f"{label} exceeds the {max_bytes}-byte input limit: {target}"
            )
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SafeInputError(
                f"{label} is not valid UTF-8 and cannot be decoded: {target}"
            ) from error
    except OSError as error:
        raise SafeInputError(f"cannot read {label} {target}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _construct_unique_mapping(
    loader: UniqueKeySafeLoader,
    node: yaml.Node,
    deep: bool = False,
) -> dict[object, object]:
    if not isinstance(node, yaml.MappingNode):
        raise yaml.constructor.ConstructorError(
            None,
            None,
            "expected a mapping node",
            node.start_mark,
        )
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def safe_load_unique(serialized: str) -> Any:
    """Load only SafeLoader types and reject ambiguous duplicate mappings."""

    return yaml.load(serialized, Loader=UniqueKeySafeLoader)


__all__ = [
    "SafeInputError",
    "UniqueKeySafeLoader",
    "read_bounded_regular_text",
    "safe_load_unique",
]
