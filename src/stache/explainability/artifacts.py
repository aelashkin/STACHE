"""Safe, versioned persistence for domain-neutral RR search results.

The search core deliberately has no persistence dependency.  This module is
the boundary that turns immutable core models into primitive-only documents
and delegates state/key meaning to a connector-owned artifact codec.
"""

from __future__ import annotations

from collections import deque
from dataclasses import fields
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import yaml

from stache.utils.safe_yaml import safe_load_unique

from .core.connector import ConnectorIdentity, MetricCertificate
from .core.models import (
    CORE_SCHEMA_VERSION,
    CounterfactualExistence,
    CounterfactualProjection,
    CounterfactualSelection,
    MinimumBasis,
    SearchCompleteness,
    SearchExtent,
    SearchMetadata,
    SearchOptions,
    SearchResult,
    SearchStats,
    StateRecord,
    StopReason,
)
from .core.policy import (
    ACTION_NORMALIZATION_SCHEMA_VERSION,
    PolicyConfigurationError,
    policy_fingerprint_from_source,
)
from .core.search import derive_search_fingerprint


ARTIFACT_SCHEMA = "stache.rr-result"
ARTIFACT_VERSION = 1


class ArtifactError(Exception):
    """Base class for safe RR artifact failures."""


class ArtifactSchemaError(ArtifactError, ValueError):
    """An artifact document is malformed or uses an unknown schema."""


class ArtifactCompatibilityError(ArtifactError, ValueError):
    """An artifact does not belong to the requested connector or policy."""


def _primitive_copy(
    value: object,
    *,
    path: str,
    error_type: type[ArtifactError],
    _active: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    """Return a detached JSON/YAML-safe value, rejecting implicit coercions."""

    if _depth > 100:
        raise error_type(f"{path} exceeds the artifact nesting limit")
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise error_type(f"{path} must contain only finite primitive values")
        return value
    if _active is None:
        _active = set()
    if isinstance(value, list):
        marker = id(value)
        if marker in _active:
            raise error_type(f"{path} contains a recursive artifact value")
        _active.add(marker)
        try:
            return [
                _primitive_copy(
                    item,
                    path=f"{path}[{index}]",
                    error_type=error_type,
                    _active=_active,
                    _depth=_depth + 1,
                )
                for index, item in enumerate(value)
            ]
        finally:
            _active.remove(marker)
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in _active:
            raise error_type(f"{path} contains a recursive artifact value")
        _active.add(marker)
        copied: dict[str, Any] = {}
        try:
            for key, item in value.items():
                if type(key) is not str:
                    raise error_type(
                        f"{path} must use primitive string keys; got {key!r}"
                    )
                copied[key] = _primitive_copy(
                    item,
                    path=f"{path}.{key}",
                    error_type=error_type,
                    _active=_active,
                    _depth=_depth + 1,
                )
            return copied
        finally:
            _active.remove(marker)
    raise error_type(
        f"{path} must contain only primitive dict/list/scalar values; "
        f"got {type(value).__name__}"
    )


def _identity_document(identity: ConnectorIdentity) -> dict[str, object]:
    return {field.name: getattr(identity, field.name) for field in fields(identity)}


def _certificate_document(certificate: MetricCertificate) -> dict[str, object]:
    return {
        field.name: getattr(certificate, field.name)
        for field in fields(certificate)
    }


def _codec(connector: object) -> object:
    try:
        codec = getattr(connector, "artifact_codec")
    except AttributeError as exc:
        raise ArtifactError(
            "connector does not expose the required artifact_codec"
        ) from exc
    for operation in ("encode_state", "decode_state", "encode_key", "decode_key"):
        if not callable(getattr(codec, operation, None)):
            raise ArtifactError(
                f"connector artifact codec is missing callable {operation}"
            )
    return codec


def _canonical_state(connector: object, state: object, *, path: str) -> object:
    try:
        canonical = connector.canonicalize(state)  # type: ignore[attr-defined]
        connector.validate_state(canonical)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ArtifactSchemaError(f"{path} is not a valid connector state: {exc}") from exc
    return canonical


def _encode_record(record: StateRecord[Any, Any], connector: object) -> dict[str, object]:
    codec = _codec(connector)
    try:
        canonical = connector.canonicalize(record.state)  # type: ignore[attr-defined]
        connector.validate_state(canonical)  # type: ignore[attr-defined]
        canonical_key = connector.state_key(canonical)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ArtifactError(f"state record is invalid for the connector: {exc}") from exc
    if canonical != record.state:
        raise ArtifactError("state codec requires canonical state records")
    if canonical_key != record.key:
        raise ArtifactError("state record key disagrees with connector state identity")
    _validate_action(
        record.action,
        connector,
        path="state record action",
        error_type=ArtifactError,
    )

    try:
        encoded_state = codec.encode_state(canonical)  # type: ignore[attr-defined]
        encoded_key = codec.encode_key(record.key)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ArtifactError(f"connector artifact codec could not encode state/key: {exc}") from exc
    encoded_state = _primitive_copy(
        encoded_state,
        path="encoded state",
        error_type=ArtifactError,
    )
    encoded_key = _primitive_copy(
        encoded_key,
        path="encoded key",
        error_type=ArtifactError,
    )

    try:
        decoded_state = codec.decode_state(encoded_state)  # type: ignore[attr-defined]
        round_trip_state = connector.canonicalize(decoded_state)  # type: ignore[attr-defined]
        connector.validate_state(round_trip_state)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ArtifactError(f"state codec round-trip failed: {exc}") from exc
    if round_trip_state != canonical:
        raise ArtifactError("state codec round-trip changed the canonical state")
    try:
        reencoded_state = _primitive_copy(
            codec.encode_state(round_trip_state),  # type: ignore[attr-defined]
            path="re-encoded state",
            error_type=ArtifactError,
        )
    except Exception as exc:
        if isinstance(exc, ArtifactError):
            raise
        raise ArtifactError(f"state codec round-trip failed: {exc}") from exc
    if reencoded_state != encoded_state:
        raise ArtifactError("state codec round-trip is not canonical")

    try:
        decoded_key = codec.decode_key(encoded_key)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ArtifactError(f"key codec round-trip failed: {exc}") from exc
    if decoded_key != record.key:
        raise ArtifactError("key codec round-trip changed the canonical key")
    try:
        reencoded_key = _primitive_copy(
            codec.encode_key(decoded_key),  # type: ignore[attr-defined]
            path="re-encoded key",
            error_type=ArtifactError,
        )
    except Exception as exc:
        if isinstance(exc, ArtifactError):
            raise
        raise ArtifactError(f"key codec round-trip failed: {exc}") from exc
    if reencoded_key != encoded_key:
        raise ArtifactError("key codec round-trip is not canonical")

    return {
        "state": encoded_state,
        "key": encoded_key,
        "action": record.action,
        "graph_depth": record.graph_depth,
        "formal_distance": record.formal_distance,
        "discovery_source": record.discovery_source,
    }


def _options_document(options: SearchOptions) -> dict[str, object]:
    return {
        "counterfactuals": options.counterfactuals.value,
        "minimum_basis": options.minimum_basis.value,
        "extent": options.extent.value,
        "max_expanded": options.max_expanded,
        "max_policy_queries": options.max_policy_queries,
        "max_graph_depth": options.max_graph_depth,
    }


def _completeness_document(value: SearchCompleteness) -> dict[str, object]:
    return {
        "region_complete": value.region_complete,
        "boundary_complete": value.boundary_complete,
        "radius_complete": value.radius_complete,
        "minimal_counterfactuals_complete": value.minimal_counterfactuals_complete,
        "max_evaluated_graph_depth": value.max_evaluated_graph_depth,
        "max_expanded_graph_depth": value.max_expanded_graph_depth,
        "max_scanned_formal_distance": value.max_scanned_formal_distance,
        "remaining_frontier_size": value.remaining_frontier_size,
        "stop_reason": _stop_reason_document(value.stop_reason),
    }


def _stats_document(value: SearchStats) -> dict[str, int]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _stop_reason_document(value: StopReason) -> str:
    """Expose a stable past-tense completion token in the artifact schema."""

    if value is StopReason.COMPLETE:
        return "completed"
    return value.value


def _continuation_document(result: SearchResult[Any, Any]) -> dict[str, object] | None:
    """Describe an in-memory checkpoint without publishing its opaque payload."""

    continuation = result.continuation
    if continuation is None:
        return None
    return {
        "resumable": False,
        "checkpoint_version": continuation.checkpoint_version,
        "fingerprint": continuation.fingerprint,
        "payload_digest": continuation.payload_digest,
        "remaining_frontier_size": result.completeness.remaining_frontier_size,
    }


def result_to_document(
    result: SearchResult[Any, Any],
    connector: object,
    *,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Convert a result to the version-1 primitive interchange document."""

    if not isinstance(result, SearchResult):
        raise TypeError("result must be a SearchResult")
    identity = getattr(connector, "identity", None)
    certificate = getattr(connector, "metric_certificate", None)
    if not isinstance(identity, ConnectorIdentity):
        raise ArtifactError("connector identity must be ConnectorIdentity")
    if not isinstance(certificate, MetricCertificate):
        raise ArtifactError("connector metric certificate must be MetricCertificate")
    if result.metadata.connector_identity != identity:
        raise ArtifactCompatibilityError("result connector identity mismatch")
    if result.metadata.metric_certificate != certificate:
        raise ArtifactCompatibilityError("result metric certificate mismatch")

    encode_many = lambda records: [  # noqa: E731 - local schema operation
        _encode_record(record, connector) for record in records
    ]
    projected_minimal = (
        None
        if result.counterfactuals.minimal is None
        else encode_many(result.counterfactuals.minimal)
    )
    projected_boundary = (
        None
        if result.counterfactuals.boundary is None
        else encode_many(result.counterfactuals.boundary)
    )
    provenance_document = _primitive_copy(
        {} if provenance is None else provenance,
        path="provenance",
        error_type=ArtifactError,
    )
    policy_source = _primitive_copy(
        result.metadata.policy_source,
        path="policy.source",
        error_type=ArtifactError,
    )
    try:
        derived_policy_fingerprint = policy_fingerprint_from_source(
            policy_source
        )
    except PolicyConfigurationError as error:
        raise ArtifactCompatibilityError(
            f"result policy source identity is invalid: {error}"
        ) from error
    if derived_policy_fingerprint != result.metadata.policy_fingerprint:
        raise ArtifactCompatibilityError(
            "result policy fingerprint disagrees with policy source identity"
        )
    derived_search_fingerprint = derive_search_fingerprint(
        connector,
        policy_fingerprint=derived_policy_fingerprint,
        options=result.metadata.options,
    )
    if derived_search_fingerprint != result.metadata.search_fingerprint:
        raise ArtifactCompatibilityError(
            "result search fingerprint disagrees with scientific identity"
        )

    document: dict[str, object] = {
        "schema": ARTIFACT_SCHEMA,
        "schema_version": ARTIFACT_VERSION,
        "connector": _identity_document(identity),
        "metric_certificate": _certificate_document(certificate),
        "policy": {
            "fingerprint": result.metadata.policy_fingerprint,
            "source": policy_source,
            "action_normalization_schema_version": (
                ACTION_NORMALIZATION_SCHEMA_VERSION
            ),
        },
        "options": _options_document(result.metadata.options),
        "metadata": {
            "search_fingerprint": result.metadata.search_fingerprint,
            "core_schema_version": result.metadata.core_schema_version,
        },
        "result": {
            "seed": _encode_record(result.seed, connector),
            "seed_action": result.seed_action,
            "region": encode_many(result.region),
            "boundary_counterfactuals": encode_many(
                result.boundary_counterfactuals
            ),
            "minimal_counterfactuals": encode_many(
                result.minimal_counterfactuals
            ),
            "counterfactuals": {
                "minimal": projected_minimal,
                "boundary": projected_boundary,
            },
            "robustness_radius": result.robustness_radius,
            "best_known_radius": result.best_known_radius,
            "counterfactual_existence": result.counterfactual_existence.value,
            "completeness": _completeness_document(result.completeness),
            "stats": _stats_document(result.stats),
            "stop_reason": _stop_reason_document(result.completeness.stop_reason),
            "continuation": _continuation_document(result),
        },
        "provenance": provenance_document,
    }
    primitive_document = _primitive_copy(
        document,
        path="document",
        error_type=ArtifactError,
    )
    # Validate writer-side invariants too, so malformed in-memory results are
    # never published merely because all their fields happen to be primitive.
    document_to_result(
        primitive_document,
        connector,
        expected_policy_fingerprint=result.metadata.policy_fingerprint,
    )
    return primitive_document


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactSchemaError(f"{path} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ArtifactSchemaError(f"{path} must use string keys")
    return value  # type: ignore[return-value]


def _required(mapping: Mapping[str, object], key: str, *, path: str) -> object:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ArtifactSchemaError(f"{path}.{key} is required") from exc


def _integer(value: object, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ArtifactSchemaError(f"{path} must be an integer >= {minimum}")
    return value


def _optional_integer(value: object, *, path: str, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _integer(value, path=path, minimum=minimum)


def _number(
    value: object,
    *,
    path: str,
    optional: bool = False,
    minimum: int | float | None = None,
) -> int | float | None:
    if value is None and optional:
        return None
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ArtifactSchemaError(f"{path} must be a finite number")
    if minimum is not None and value < minimum:
        if minimum == 0:
            raise ArtifactSchemaError(f"{path} must be a non-negative number")
        raise ArtifactSchemaError(f"{path} must be a number >= {minimum}")
    return value  # type: ignore[return-value]


def _boolean(value: object, *, path: str) -> bool:
    if type(value) is not bool:
        raise ArtifactSchemaError(f"{path} must be a boolean")
    return value


def _string(value: object, *, path: str) -> str:
    if type(value) is not str or not value:
        raise ArtifactSchemaError(f"{path} must be a non-empty string")
    return value


def _action_count(
    connector: object,
    *,
    error_type: type[ArtifactError],
) -> int:
    action_spec = getattr(connector, "action_spec", None)
    count = getattr(action_spec, "count", None)
    if type(count) is not int or count <= 0:
        raise error_type(
            "connector action_spec.count must be a positive integer for artifacts"
        )
    return count


def _validate_action(
    value: object,
    connector: object,
    *,
    path: str,
    error_type: type[ArtifactError],
) -> int:
    count = _action_count(connector, error_type=error_type)
    if type(value) is not int or value < 0 or value >= count:
        raise error_type(
            f"{path} is outside the connector action range [0, {count}): {value!r}"
        )
    return value


def _enum(value: object, enum_type: type[Any], *, path: str) -> Any:
    raw = _string(value, path=path)
    try:
        return enum_type(raw)
    except ValueError as exc:
        raise ArtifactSchemaError(f"{path} has unknown value {raw!r}") from exc


def _decode_stop_reason(value: object, *, path: str) -> StopReason:
    raw = _string(value, path=path)
    if raw == "completed":
        return StopReason.COMPLETE
    try:
        return StopReason(raw)
    except ValueError as exc:
        raise ArtifactSchemaError(f"{path} has unknown value {raw!r}") from exc


def _validate_connector(document: Mapping[str, object], connector: object) -> None:
    identity = getattr(connector, "identity", None)
    certificate = getattr(connector, "metric_certificate", None)
    if not isinstance(identity, ConnectorIdentity):
        raise ArtifactCompatibilityError("connector identity is unavailable")
    if not isinstance(certificate, MetricCertificate):
        raise ArtifactCompatibilityError("connector metric certificate is unavailable")

    encoded_identity = _mapping(
        _required(document, "connector", path="document"),
        path="connector",
    )
    for field in fields(identity):
        actual = _required(encoded_identity, field.name, path="connector")
        expected = getattr(identity, field.name)
        if actual != expected:
            raise ArtifactCompatibilityError(
                f"connector {field.name} mismatch: artifact={actual!r}, "
                f"requested={expected!r}"
            )

    encoded_certificate = _mapping(
        _required(document, "metric_certificate", path="document"),
        path="metric_certificate",
    )
    for field in fields(certificate):
        actual = _required(
            encoded_certificate,
            field.name,
            path="metric_certificate",
        )
        expected = getattr(certificate, field.name)
        if actual != expected or type(actual) is not type(expected):
            raise ArtifactCompatibilityError(
                f"metric certificate {field.name} mismatch: artifact={actual!r}, "
                f"requested={expected!r}"
            )


def _decode_record(value: object, connector: object, *, path: str) -> StateRecord[Any, Any]:
    record = _mapping(value, path=path)
    encoded_state = _required(record, "state", path=path)
    encoded_key = _required(record, "key", path=path)
    codec = _codec(connector)
    try:
        decoded_state = codec.decode_state(encoded_state)  # type: ignore[attr-defined]
        state = connector.canonicalize(decoded_state)  # type: ignore[attr-defined]
        connector.validate_state(state)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ArtifactSchemaError(f"{path}.state could not be decoded: {exc}") from exc
    try:
        key = codec.decode_key(encoded_key)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ArtifactSchemaError(f"{path}.key could not be decoded: {exc}") from exc
    try:
        expected_key = connector.state_key(state)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ArtifactSchemaError(f"{path}.state has no valid key: {exc}") from exc
    if key != expected_key:
        raise ArtifactSchemaError(
            f"{path} key disagrees with its decoded state identity"
        )

    try:
        canonical_state_encoding = _primitive_copy(
            codec.encode_state(state),  # type: ignore[attr-defined]
            path=f"{path}.state",
            error_type=ArtifactSchemaError,
        )
        canonical_key_encoding = _primitive_copy(
            codec.encode_key(key),  # type: ignore[attr-defined]
            path=f"{path}.key",
            error_type=ArtifactSchemaError,
        )
    except Exception as exc:
        if isinstance(exc, ArtifactSchemaError):
            raise
        raise ArtifactSchemaError(f"{path} codec round-trip failed: {exc}") from exc
    if canonical_state_encoding != encoded_state:
        raise ArtifactSchemaError(f"{path}.state encoding is not canonical")
    if canonical_key_encoding != encoded_key:
        raise ArtifactSchemaError(f"{path}.key encoding is not canonical")

    action = _validate_action(
        _required(record, "action", path=path),
        connector,
        path=f"{path}.action",
        error_type=ArtifactSchemaError,
    )
    graph_depth = _optional_integer(
        _required(record, "graph_depth", path=path),
        path=f"{path}.graph_depth",
    )
    formal_distance = _number(
        _required(record, "formal_distance", path=path),
        path=f"{path}.formal_distance",
        minimum=0,
    )
    discovery_source = _string(
        _required(record, "discovery_source", path=path),
        path=f"{path}.discovery_source",
    )
    return StateRecord(
        state=state,
        key=key,
        action=action,
        graph_depth=graph_depth,
        formal_distance=formal_distance,
        discovery_source=discovery_source,
    )


def _decode_records(
    value: object,
    connector: object,
    *,
    path: str,
) -> tuple[StateRecord[Any, Any], ...]:
    if not isinstance(value, list):
        raise ArtifactSchemaError(f"{path} must be a list")
    return tuple(
        _decode_record(item, connector, path=f"{path}[{index}]")
        for index, item in enumerate(value)
    )


def _decode_projected(
    value: object,
    connector: object,
    *,
    path: str,
) -> tuple[StateRecord[Any, Any], ...] | None:
    if value is None:
        return None
    return _decode_records(value, connector, path=path)


def _validate_continuation_summary(
    value: object,
    *,
    completeness: SearchCompleteness,
    search_fingerprint: str,
) -> None:
    budget_reasons = {
        StopReason.MAX_EXPANDED,
        StopReason.MAX_POLICY_QUERIES,
        StopReason.MAX_GRAPH_DEPTH,
    }
    if value is None:
        if completeness.stop_reason in budget_reasons:
            raise ArtifactSchemaError(
                "result.continuation summary is required for a budget stop"
            )
        if completeness.remaining_frontier_size != 0:
            raise ArtifactSchemaError(
                "result.continuation summary is required for a remaining frontier"
            )
        return
    if completeness.stop_reason not in budget_reasons:
        raise ArtifactSchemaError(
            "result.continuation summary is only valid for a budget stop"
        )
    summary = _mapping(value, path="result.continuation")
    resumable = _required(summary, "resumable", path="result.continuation")
    if resumable is not False:
        raise ArtifactSchemaError(
            "result.continuation.resumable must be false for artifact version 1"
        )
    _string(
        _required(summary, "checkpoint_version", path="result.continuation"),
        path="result.continuation.checkpoint_version",
    )
    fingerprint = _string(
        _required(summary, "fingerprint", path="result.continuation"),
        path="result.continuation.fingerprint",
    )
    if fingerprint != search_fingerprint:
        raise ArtifactSchemaError(
            "result.continuation fingerprint disagrees with metadata"
        )
    digest = _string(
        _required(summary, "payload_digest", path="result.continuation"),
        path="result.continuation.payload_digest",
    )
    if (
        not digest.startswith("sha256:")
        or len(digest) != len("sha256:") + 64
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise ArtifactSchemaError(
            "result.continuation.payload_digest must be a sha256 digest"
        )
    remaining = _integer(
        _required(summary, "remaining_frontier_size", path="result.continuation"),
        path="result.continuation.remaining_frontier_size",
    )
    if remaining != completeness.remaining_frontier_size:
        raise ArtifactSchemaError(
            "result.continuation frontier size disagrees with completeness"
        )


def _decode_options(value: object) -> SearchOptions:
    options = _mapping(value, path="options")
    try:
        return SearchOptions(
            counterfactuals=_enum(
                _required(options, "counterfactuals", path="options"),
                CounterfactualSelection,
                path="options.counterfactuals",
            ),
            minimum_basis=_enum(
                _required(options, "minimum_basis", path="options"),
                MinimumBasis,
                path="options.minimum_basis",
            ),
            extent=_enum(
                _required(options, "extent", path="options"),
                SearchExtent,
                path="options.extent",
            ),
            max_expanded=_optional_integer(
                _required(options, "max_expanded", path="options"),
                path="options.max_expanded",
            ),
            max_policy_queries=_optional_integer(
                _required(options, "max_policy_queries", path="options"),
                path="options.max_policy_queries",
                minimum=1,
            ),
            max_graph_depth=_optional_integer(
                _required(options, "max_graph_depth", path="options"),
                path="options.max_graph_depth",
            ),
        )
    except ValueError as exc:
        if isinstance(exc, ArtifactSchemaError):
            raise
        raise ArtifactSchemaError(f"options are invalid: {exc}") from exc


def _decode_completeness(value: object) -> SearchCompleteness:
    item = _mapping(value, path="result.completeness")
    return SearchCompleteness(
        region_complete=_boolean(
            _required(item, "region_complete", path="result.completeness"),
            path="result.completeness.region_complete",
        ),
        boundary_complete=_boolean(
            _required(item, "boundary_complete", path="result.completeness"),
            path="result.completeness.boundary_complete",
        ),
        radius_complete=_boolean(
            _required(item, "radius_complete", path="result.completeness"),
            path="result.completeness.radius_complete",
        ),
        minimal_counterfactuals_complete=_boolean(
            _required(
                item,
                "minimal_counterfactuals_complete",
                path="result.completeness",
            ),
            path="result.completeness.minimal_counterfactuals_complete",
        ),
        max_evaluated_graph_depth=_integer(
            _required(item, "max_evaluated_graph_depth", path="result.completeness"),
            path="result.completeness.max_evaluated_graph_depth",
        ),
        max_expanded_graph_depth=_optional_integer(
            _required(item, "max_expanded_graph_depth", path="result.completeness"),
            path="result.completeness.max_expanded_graph_depth",
        ),
        max_scanned_formal_distance=_number(
            _required(item, "max_scanned_formal_distance", path="result.completeness"),
            path="result.completeness.max_scanned_formal_distance",
            optional=True,
            minimum=0,
        ),
        remaining_frontier_size=_integer(
            _required(item, "remaining_frontier_size", path="result.completeness"),
            path="result.completeness.remaining_frontier_size",
        ),
        stop_reason=_decode_stop_reason(
            _required(item, "stop_reason", path="result.completeness"),
            path="result.completeness.stop_reason",
        ),
    )


def _decode_stats(value: object) -> SearchStats:
    item = _mapping(value, path="result.stats")
    values = {
        field.name: _integer(
            _required(item, field.name, path="result.stats"),
            path=f"result.stats.{field.name}",
        )
        for field in fields(SearchStats)
    }
    return SearchStats(**values)


def _unique_record_keys(
    records: tuple[StateRecord[Any, Any], ...],
    *,
    path: str,
) -> set[Any]:
    keys: set[Any] = set()
    for record in records:
        try:
            duplicate = record.key in keys
        except TypeError as exc:
            raise ArtifactSchemaError(f"{path} contains an unhashable record key") from exc
        if duplicate:
            raise ArtifactSchemaError(
                f"{path} contains duplicate record key {record.key!r}"
            )
        keys.add(record.key)
    return keys


def _connector_formal_distance(
    connector: object,
    seed: object,
    state: object,
    *,
    path: str,
) -> int | float:
    try:
        value = connector.formal_distance(seed, state)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ArtifactSchemaError(
            f"{path} formal distance could not be recomputed by the connector: {exc}"
        ) from exc
    if (
        isinstance(value, bool)
        or type(value) not in {int, float}
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ArtifactSchemaError(
            f"{path} connector formal distance must be finite and non-negative"
        )
    return value


def _validate_record_integrity(
    *,
    connector: object,
    seed: StateRecord[Any, Any],
    region: tuple[StateRecord[Any, Any], ...],
    boundary: tuple[StateRecord[Any, Any], ...],
    minima: tuple[StateRecord[Any, Any], ...],
) -> None:
    if seed.graph_depth != 0:
        raise ArtifactSchemaError("result seed graph depth must be zero")
    if seed.formal_distance != 0:
        raise ArtifactSchemaError("result seed formal distance must be zero")
    if seed.discovery_source != "graph":
        raise ArtifactSchemaError("result seed discovery source must be graph")

    groups = (
        ("result.seed", (seed,), False),
        ("result.region", region, True),
        ("result.boundary_counterfactuals", boundary, True),
        ("result.minimal_counterfactuals", minima, False),
    )
    known: dict[Any, tuple[StateRecord[Any, Any], str]] = {}
    for path, records, graph_only in groups:
        for index, record in enumerate(records):
            record_path = path if path == "result.seed" else f"{path}[{index}]"
            if record.discovery_source not in {"graph", "formal"}:
                raise ArtifactSchemaError(
                    f"{record_path}.discovery_source must be graph or formal"
                )
            if graph_only and record.discovery_source != "graph":
                raise ArtifactSchemaError(
                    f"{record_path}.discovery_source must be graph"
                )
            if record.discovery_source == "graph" and record.graph_depth is None:
                raise ArtifactSchemaError(
                    f"{record_path} graph discovery requires a graph depth"
                )
            if record.discovery_source == "formal" and record.graph_depth is not None:
                raise ArtifactSchemaError(
                    f"{record_path} formal-only discovery may not have a graph depth"
                )

            expected_distance = _connector_formal_distance(
                connector,
                seed.state,
                record.state,
                path=record_path,
            )
            if record.formal_distance != expected_distance:
                raise ArtifactSchemaError(
                    f"{record_path}.formal_distance disagrees with the connector: "
                    f"artifact={record.formal_distance!r}, "
                    f"connector={expected_distance!r}"
                )

            previous = known.get(record.key)
            if previous is None:
                known[record.key] = (record, record_path)
                continue
            existing, existing_path = previous
            if (
                record.state != existing.state
                or record.action != existing.action
                or record.graph_depth != existing.graph_depth
                or record.formal_distance != existing.formal_distance
            ):
                raise ArtifactSchemaError(
                    "conflicting record representations for state key "
                    f"{record.key!r}: {existing_path} and {record_path}"
                )


def _validate_complete_graph_evidence(
    *,
    connector: object,
    seed: StateRecord[Any, Any],
    region: tuple[StateRecord[Any, Any], ...],
    boundary: tuple[StateRecord[Any, Any], ...],
) -> None:
    region_by_key = {record.key: record for record in region}
    boundary_by_key = {record.key: record for record in boundary}
    known = {**region_by_key, **boundary_by_key}
    adjacency: dict[Any, set[Any]] = {}

    for record in region:
        neighbor_keys: set[Any] = set()
        try:
            raw_neighbors = connector.atomic_neighbors(record.state)  # type: ignore[attr-defined]
            for raw_neighbor in raw_neighbors:
                canonical = connector.canonicalize(raw_neighbor)  # type: ignore[attr-defined]
                connector.validate_state(canonical)  # type: ignore[attr-defined]
                key = connector.state_key(canonical)  # type: ignore[attr-defined]
                if key == record.key:
                    raise ArtifactSchemaError(
                        f"connector returned a self-neighbor for {record.key!r}"
                    )
                if key in neighbor_keys:
                    raise ArtifactSchemaError(
                        f"connector returned duplicate neighbor key {key!r}"
                    )
                neighbor_keys.add(key)
                known_record = known.get(key)
                if known_record is None:
                    raise ArtifactSchemaError(
                        "complete graph result omits a connector neighbor of "
                        f"region state {record.key!r}: {key!r}"
                    )
                if known_record.state != canonical:
                    raise ArtifactSchemaError(
                        f"connector neighbor state conflicts for key {key!r}"
                    )
        except ArtifactSchemaError:
            raise
        except Exception as exc:
            raise ArtifactSchemaError(
                f"could not validate graph neighbors for {record.key!r}: {exc}"
            ) from exc
        adjacency[record.key] = neighbor_keys

    expected_depths: dict[Any, int] = {seed.key: 0}
    frontier: deque[Any] = deque([seed.key])
    while frontier:
        parent = frontier.popleft()
        for key in adjacency[parent]:
            if key in expected_depths:
                continue
            expected_depths[key] = expected_depths[parent] + 1
            if key in region_by_key:
                frontier.append(key)

    missing = set(known) - set(expected_depths)
    if missing:
        raise ArtifactSchemaError(
            "complete graph records are not reachable by RR BFS: "
            f"{len(missing)} state(s)"
        )
    for key, record in known.items():
        if record.graph_depth != expected_depths[key]:
            raise ArtifactSchemaError(
                "record graph depth disagrees with connector RR BFS for "
                f"{key!r}: artifact={record.graph_depth!r}, "
                f"expected={expected_depths[key]}"
            )


def _validate_result_invariants(
    *,
    connector: object,
    seed: StateRecord[Any, Any],
    seed_action: int,
    region: tuple[StateRecord[Any, Any], ...],
    boundary: tuple[StateRecord[Any, Any], ...],
    minima: tuple[StateRecord[Any, Any], ...],
    options: SearchOptions,
    robustness_radius: int | float | None,
    best_known_radius: int | float | None,
    existence: CounterfactualExistence,
    completeness: SearchCompleteness,
    stats: SearchStats,
) -> None:
    region_keys = _unique_record_keys(region, path="result.region")
    boundary_keys = _unique_record_keys(
        boundary,
        path="result.boundary_counterfactuals",
    )
    minima_keys = _unique_record_keys(
        minima,
        path="result.minimal_counterfactuals",
    )

    _validate_record_integrity(
        connector=connector,
        seed=seed,
        region=region,
        boundary=boundary,
        minima=minima,
    )

    if seed.key not in region_keys or not any(record == seed for record in region):
        raise ArtifactSchemaError("result seed must be a member of the region")
    if region_keys.intersection(boundary_keys):
        raise ArtifactSchemaError(
            "result region and boundary must use disjoint record keys"
        )
    if any(record.action != seed_action for record in region):
        raise ArtifactSchemaError(
            "result region actions must equal the normalized seed action"
        )
    if any(record.action == seed_action for record in boundary):
        raise ArtifactSchemaError(
            "result boundary actions must differ from the seed action"
        )
    if any(record.action == seed_action for record in minima):
        raise ArtifactSchemaError(
            "result minimal counterfactual actions must differ from the seed action"
        )
    if (
        options.minimum_basis is MinimumBasis.GRAPH_BOUNDARY
        and not minima_keys.issubset(boundary_keys)
    ):
        raise ArtifactSchemaError(
            "graph-boundary minima must be members of the boundary"
        )

    if (
        completeness.minimal_counterfactuals_complete
        and not completeness.radius_complete
    ):
        raise ArtifactSchemaError(
            "minimal-counterfactual completeness requires radius completeness"
        )
    if completeness.boundary_complete and not completeness.region_complete:
        raise ArtifactSchemaError(
            "boundary completeness requires region completeness"
        )
    if completeness.stop_reason is StopReason.COMPLETE:
        if not (
            completeness.region_complete and completeness.boundary_complete
        ):
            raise ArtifactSchemaError(
                "completed traversal requires region and boundary completeness"
            )
        if completeness.remaining_frontier_size != 0:
            raise ArtifactSchemaError(
                "completed results may not retain a remaining frontier"
            )
    if completeness.stop_reason is StopReason.THROUGH_MINIMAL:
        if not (
            completeness.radius_complete
            and completeness.minimal_counterfactuals_complete
        ):
            raise ArtifactSchemaError(
                "through-minimal results require complete radius and tied minima"
            )
        if completeness.remaining_frontier_size != 0:
            raise ArtifactSchemaError(
                "through-minimal results may not retain a remaining frontier"
            )

    has_counterfactual_record = bool(boundary or minima)
    if existence is CounterfactualExistence.FOUND:
        if not has_counterfactual_record:
            raise ArtifactSchemaError(
                "counterfactual existence is found but no counterfactual record exists"
            )
        if best_known_radius is None:
            raise ArtifactSchemaError(
                "found counterfactuals require a best-known radius"
            )
        if completeness.radius_complete and robustness_radius is None:
            raise ArtifactSchemaError(
                "found counterfactuals with a complete radius require robustness_radius"
            )
        if completeness.minimal_counterfactuals_complete and not minima:
            raise ArtifactSchemaError(
                "found counterfactuals with complete ties require minimal records"
            )
    elif existence is CounterfactualExistence.PROVEN_ABSENT:
        if has_counterfactual_record:
            raise ArtifactSchemaError(
                "proven-absent counterfactual existence conflicts with records"
            )
        if not (
            completeness.radius_complete
            and completeness.minimal_counterfactuals_complete
        ):
            raise ArtifactSchemaError(
                "proven absence requires complete radius and tied minima"
            )
        if robustness_radius is not None or best_known_radius is not None:
            raise ArtifactSchemaError(
                "proven absence requires null radius values"
            )
    else:
        if has_counterfactual_record:
            raise ArtifactSchemaError(
                "unknown counterfactual existence conflicts with found records"
            )
        if robustness_radius is not None:
            raise ArtifactSchemaError(
                "unknown counterfactual existence requires a null robustness radius"
            )
        if best_known_radius is not None:
            raise ArtifactSchemaError(
                "unknown counterfactual existence requires a null best-known radius"
            )

    if completeness.region_complete and completeness.boundary_complete:
        _validate_complete_graph_evidence(
            connector=connector,
            seed=seed,
            region=region,
            boundary=boundary,
        )

    if not completeness.radius_complete and robustness_radius is not None:
        raise ArtifactSchemaError(
            "an incomplete radius requires a null robustness_radius"
        )
    if robustness_radius is not None and best_known_radius != robustness_radius:
        raise ArtifactSchemaError(
            "a certified robustness radius must equal the best-known radius"
        )

    known_counterfactuals = {
        record.key: record for record in (*boundary, *minima)
    }
    if existence is CounterfactualExistence.FOUND:
        if options.minimum_basis is MinimumBasis.GRAPH_BOUNDARY:
            observed_radii = [
                float(record.graph_depth)
                for record in boundary
                if record.graph_depth is not None
            ]
        else:
            observed_radii = [
                float(record.formal_distance)
                for record in known_counterfactuals.values()
            ]
        if not observed_radii:
            raise ArtifactSchemaError(
                "found counterfactuals have no observed radius evidence"
            )
        observed_radius = min(observed_radii)
        if best_known_radius is None or float(best_known_radius) != observed_radius:
            raise ArtifactSchemaError(
                "best-known radius disagrees with observed counterfactual records"
            )
        if (
            completeness.radius_complete
            and (
                robustness_radius is None
                or float(robustness_radius) != observed_radius
            )
        ):
            raise ArtifactSchemaError(
                "certified radius is not the minimum observed counterfactual radius"
            )
    if robustness_radius is not None and minima:
        if options.minimum_basis is MinimumBasis.GRAPH_BOUNDARY:
            if any(
                record.graph_depth is None
                or float(record.graph_depth) != float(robustness_radius)
                for record in minima
            ):
                raise ArtifactSchemaError(
                    "graph-boundary minima must match the robustness radius"
                )
        elif any(
            float(record.formal_distance) != float(robustness_radius)
            for record in minima
        ):
            raise ArtifactSchemaError(
                "formal-global minima must match the robustness radius"
            )

    if (
        completeness.minimal_counterfactuals_complete
        and robustness_radius is not None
    ):
        if options.minimum_basis is MinimumBasis.GRAPH_BOUNDARY:
            expected_minima = {
                record.key
                for record in boundary
                if record.graph_depth is not None
                and float(record.graph_depth) == float(robustness_radius)
            }
        else:
            expected_minima = {
                record.key
                for record in (*boundary, *minima)
                if float(record.formal_distance) == float(robustness_radius)
            }
        if minima_keys != expected_minima:
            raise ArtifactSchemaError(
                "complete tied minima disagree with counterfactual records at "
                "the certified robustness radius"
            )

    if stats.states_discovered < 1 or stats.states_evaluated < 1:
        raise ArtifactSchemaError("stats must include the discovered/evaluated seed")
    if stats.policy_queries < 1:
        raise ArtifactSchemaError("stats.policy_queries must include the seed query")
    if stats.states_expanded > stats.states_evaluated:
        raise ArtifactSchemaError(
            "stats.states_expanded may not exceed states_evaluated"
        )
    if stats.states_expanded > len(region):
        raise ArtifactSchemaError(
            "stats.states_expanded may not exceed known region records"
        )
    if stats.policy_queries > stats.states_evaluated:
        raise ArtifactSchemaError(
            "stats.policy_queries may not exceed states_evaluated"
        )
    if stats.table_hits > stats.policy_queries or stats.model_queries > stats.policy_queries:
        raise ArtifactSchemaError(
            "stats source-query counts may not exceed policy_queries"
        )
    if stats.table_hits + stats.model_queries > stats.policy_queries:
        raise ArtifactSchemaError(
            "stats table/model query counts may not exceed policy_queries"
        )
    known_graph_records = len(region_keys.union(boundary_keys))
    if stats.states_discovered < known_graph_records:
        raise ArtifactSchemaError(
            "stats.states_discovered is below the serialized graph record count"
        )
    if stats.states_evaluated < known_graph_records:
        raise ArtifactSchemaError(
            "stats.states_evaluated is below the serialized graph record count"
        )


def document_to_result(
    document: Mapping[str, object],
    connector: object,
    *,
    expected_policy_fingerprint: str | None = None,
) -> SearchResult[Any, Any]:
    """Validate and reconstruct an immutable result without mutating input."""

    _primitive_copy(document, path="document", error_type=ArtifactSchemaError)
    root = _mapping(document, path="document")
    schema = _required(root, "schema", path="document")
    if schema != ARTIFACT_SCHEMA:
        raise ArtifactSchemaError(
            f"schema name mismatch: expected {ARTIFACT_SCHEMA!r}, got {schema!r}"
        )
    version = _required(root, "schema_version", path="document")
    if type(version) is not int or version != ARTIFACT_VERSION:
        raise ArtifactSchemaError(
            f"schema_version mismatch: expected {ARTIFACT_VERSION}, got {version!r}"
        )
    _validate_connector(root, connector)

    policy = _mapping(_required(root, "policy", path="document"), path="policy")
    policy_fingerprint = _string(
        _required(policy, "fingerprint", path="policy"),
        path="policy.fingerprint",
    )
    action_normalization_version = _required(
        policy,
        "action_normalization_schema_version",
        path="policy",
    )
    if (
        type(action_normalization_version) is not int
        or action_normalization_version != ACTION_NORMALIZATION_SCHEMA_VERSION
    ):
        raise ArtifactCompatibilityError(
            "action normalization schema version mismatch: "
            f"expected {ACTION_NORMALIZATION_SCHEMA_VERSION}, "
            f"got {action_normalization_version!r}"
        )
    policy_source = _primitive_copy(
        _mapping(_required(policy, "source", path="policy"), path="policy.source"),
        path="policy.source",
        error_type=ArtifactSchemaError,
    )
    try:
        derived_policy_fingerprint = policy_fingerprint_from_source(
            policy_source
        )
    except PolicyConfigurationError as error:
        raise ArtifactCompatibilityError(
            f"policy source identity is invalid: {error}"
        ) from error
    if policy_fingerprint != derived_policy_fingerprint:
        raise ArtifactCompatibilityError(
            "policy fingerprint disagrees with policy source identity"
        )
    if (
        expected_policy_fingerprint is not None
        and derived_policy_fingerprint != expected_policy_fingerprint
    ):
        raise ArtifactCompatibilityError(
            "policy fingerprint mismatch: "
            f"artifact={derived_policy_fingerprint!r}, "
            f"expected={expected_policy_fingerprint!r}"
        )
    options = _decode_options(_required(root, "options", path="document"))
    metadata_document = _mapping(
        _required(root, "metadata", path="document"),
        path="metadata",
    )
    search_fingerprint = _string(
        _required(metadata_document, "search_fingerprint", path="metadata"),
        path="metadata.search_fingerprint",
    )
    core_schema_version = _integer(
        _required(metadata_document, "core_schema_version", path="metadata"),
        path="metadata.core_schema_version",
        minimum=1,
    )
    if core_schema_version != CORE_SCHEMA_VERSION:
        raise ArtifactSchemaError(
            "core_schema_version mismatch: "
            f"expected {CORE_SCHEMA_VERSION}, got {core_schema_version}"
        )
    derived_search_fingerprint = derive_search_fingerprint(
        connector,
        policy_fingerprint=derived_policy_fingerprint,
        options=options,
    )
    if search_fingerprint != derived_search_fingerprint:
        raise ArtifactCompatibilityError(
            "search fingerprint disagrees with connector, policy, or options identity"
        )

    result_document = _mapping(
        _required(root, "result", path="document"),
        path="result",
    )
    seed = _decode_record(
        _required(result_document, "seed", path="result"),
        connector,
        path="result.seed",
    )
    seed_action = _validate_action(
        _required(result_document, "seed_action", path="result"),
        connector,
        path="result.seed_action",
        error_type=ArtifactSchemaError,
    )
    if seed_action != seed.action:
        raise ArtifactSchemaError("result seed action disagrees with seed record")
    region = _decode_records(
        _required(result_document, "region", path="result"),
        connector,
        path="result.region",
    )
    boundary = _decode_records(
        _required(result_document, "boundary_counterfactuals", path="result"),
        connector,
        path="result.boundary_counterfactuals",
    )
    minima = _decode_records(
        _required(result_document, "minimal_counterfactuals", path="result"),
        connector,
        path="result.minimal_counterfactuals",
    )
    projection_document = _mapping(
        _required(result_document, "counterfactuals", path="result"),
        path="result.counterfactuals",
    )
    projected_minimal = _decode_projected(
        _required(projection_document, "minimal", path="result.counterfactuals"),
        connector,
        path="result.counterfactuals.minimal",
    )
    projected_boundary = _decode_projected(
        _required(projection_document, "boundary", path="result.counterfactuals"),
        connector,
        path="result.counterfactuals.boundary",
    )
    if projected_minimal is not None and projected_minimal != minima:
        raise ArtifactSchemaError(
            "projected minimal counterfactuals disagree with the complete result field"
        )
    if projected_boundary is not None and projected_boundary != boundary:
        raise ArtifactSchemaError(
            "projected boundary counterfactuals disagree with the complete result field"
        )

    completeness = _decode_completeness(
        _required(result_document, "completeness", path="result")
    )
    explicit_stop = _decode_stop_reason(
        _required(result_document, "stop_reason", path="result"),
        path="result.stop_reason",
    )
    if explicit_stop is not completeness.stop_reason:
        raise ArtifactSchemaError(
            "result.stop_reason disagrees with result.completeness.stop_reason"
        )
    _validate_continuation_summary(
        _required(result_document, "continuation", path="result"),
        completeness=completeness,
        search_fingerprint=search_fingerprint,
    )
    stats = _decode_stats(_required(result_document, "stats", path="result"))
    existence = _enum(
        _required(result_document, "counterfactual_existence", path="result"),
        CounterfactualExistence,
        path="result.counterfactual_existence",
    )
    robustness_radius = _number(
        _required(result_document, "robustness_radius", path="result"),
        path="result.robustness_radius",
        optional=True,
        minimum=0,
    )
    best_known_radius = _number(
        _required(result_document, "best_known_radius", path="result"),
        path="result.best_known_radius",
        optional=True,
        minimum=0,
    )

    _validate_result_invariants(
        connector=connector,
        seed=seed,
        seed_action=seed_action,
        region=region,
        boundary=boundary,
        minima=minima,
        options=options,
        robustness_radius=robustness_radius,
        best_known_radius=best_known_radius,
        existence=existence,
        completeness=completeness,
        stats=stats,
    )

    if options.counterfactuals is CounterfactualSelection.MINIMAL:
        expected_projection = CounterfactualProjection(minimal=minima, boundary=None)
    elif options.counterfactuals is CounterfactualSelection.BOUNDARY:
        expected_projection = CounterfactualProjection(minimal=None, boundary=boundary)
    else:
        expected_projection = CounterfactualProjection(minimal=minima, boundary=boundary)
    actual_projection = CounterfactualProjection(
        minimal=projected_minimal,
        boundary=projected_boundary,
    )
    if actual_projection != expected_projection:
        raise ArtifactSchemaError(
            "result counterfactual projection disagrees with options.counterfactuals"
        )

    identity = connector.identity  # type: ignore[attr-defined]
    certificate = connector.metric_certificate  # type: ignore[attr-defined]
    metadata = SearchMetadata(
        connector_identity=identity,
        metric_certificate=certificate,
        options=options,
        policy_fingerprint=policy_fingerprint,
        policy_source=policy_source,
        search_fingerprint=search_fingerprint,
        core_schema_version=core_schema_version,
    )
    return SearchResult(
        seed=seed,
        seed_action=seed_action,
        region=region,
        boundary_counterfactuals=boundary,
        minimal_counterfactuals=minima,
        counterfactuals=actual_projection,
        robustness_radius=robustness_radius,
        best_known_radius=best_known_radius,
        counterfactual_existence=existence,
        completeness=completeness,
        stats=stats,
        metadata=metadata,
        continuation=None,
    )


def save_result(
    path: str | os.PathLike[str],
    result: SearchResult[Any, Any],
    connector: object,
    *,
    provenance: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> None:
    """Atomically publish safe YAML, requiring opt-in for replacement."""

    target = Path(path)
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean")
    if not overwrite and os.path.lexists(target):
        raise ArtifactError(f"artifact already exists; overwrite refused: {target}")
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise ArtifactError(f"artifact parent directory does not exist: {parent}")

    document = result_to_document(result, connector, provenance=provenance)
    try:
        serialized = yaml.safe_dump(
            document,
            allow_unicode=True,
            sort_keys=False,
        )
    except Exception as exc:
        raise ArtifactError(f"artifact serialization interruption: {exc}") from exc

    temporary_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            dir=parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(raw_path)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, target)
            temporary_path = None
        else:
            try:
                # Hard-linking is an atomic, no-replace publish on the same
                # filesystem.  It closes the preflight exists-check race.
                os.link(temporary_path, target)
            except FileExistsError as exc:
                raise ArtifactError(
                    f"artifact already exists; overwrite refused: {target}"
                ) from exc
            temporary_path.unlink()
            temporary_path = None
        _fsync_directory(parent)
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"artifact write failed: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _fsync_directory(path: Path) -> None:
    """Durably record the atomic directory-entry update where supported."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_result(
    path: str | os.PathLike[str],
    connector: object,
    *,
    expected_policy_fingerprint: str | None = None,
) -> SearchResult[Any, Any]:
    """Safely load a versioned YAML result for a connector and policy."""

    target = Path(path)
    try:
        serialized = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"could not read artifact {target}: {exc}") from exc
    try:
        document = safe_load_unique(serialized)
    except yaml.YAMLError as exc:
        raise ArtifactSchemaError(
            f"unsafe or malformed YAML tag/construct in {target}: {exc}"
        ) from exc
    if not isinstance(document, Mapping):
        raise ArtifactSchemaError("YAML artifact root must be a mapping")
    return document_to_result(
        document,
        connector,
        expected_policy_fingerprint=expected_policy_fingerprint,
    )


__all__ = [
    "ACTION_NORMALIZATION_SCHEMA_VERSION",
    "ARTIFACT_SCHEMA",
    "ARTIFACT_VERSION",
    "ArtifactCompatibilityError",
    "ArtifactError",
    "ArtifactSchemaError",
    "document_to_result",
    "load_result",
    "result_to_document",
    "save_result",
]
