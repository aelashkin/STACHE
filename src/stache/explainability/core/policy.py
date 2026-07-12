"""Validated, cached scalar-discrete policy action sources.

The implementation uses structural model/space checks and therefore does not
import Gymnasium or Stable-Baselines3.  Table and model queries share one
canonical-state cache, including in the explicit table-then-model source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral
from typing import Hashable, Mapping, Protocol, Sequence

import numpy as np

from .connector import DiscreteActionSpec, ObservationIdentity
from .identity import IdentityEncodingError, fingerprint_document


ACTION_NORMALIZATION_SCHEMA_VERSION = 1
MODEL_MANIFEST_SCHEMA_VERSION = 1


class PolicyError(Exception):
    """Base class for action-source contract failures."""


class ActionValidationError(PolicyError, ValueError):
    """An action is not an in-range integer scalar."""


class ActionShapeError(ActionValidationError):
    """An action container does not represent exactly one scalar action."""


class UnknownTableKeyError(PolicyError, KeyError):
    """A strict table source has no action for the connector lookup key."""


class ModelCompatibilityError(PolicyError, ValueError):
    """A model or encoded observation violates the connector declaration."""


class CacheRestoreError(PolicyError, ValueError):
    """A serialized/in-memory action-cache checkpoint is malformed."""


class PolicyConfigurationError(PolicyError, ValueError):
    """An action source cannot be constructed safely."""


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Result-affecting model semantics recorded alongside model bytes."""

    model_fingerprint: str
    observation_identity: ObservationIdentity
    action_spec: DiscreteActionSpec
    schema_version: int = MODEL_MANIFEST_SCHEMA_VERSION


def model_manifest_to_document(manifest: ModelManifest) -> dict[str, object]:
    """Encode a validated manifest as primitive versioned data."""

    if not isinstance(manifest, ModelManifest):
        raise PolicyConfigurationError("manifest must be a ModelManifest")
    return asdict(manifest)


def model_manifest_from_document(document: Mapping[str, object]) -> ModelManifest:
    """Decode strict primitive manifest data without trusting caller types."""

    if not isinstance(document, Mapping):
        raise PolicyConfigurationError("model manifest must be a mapping")
    required = {
        "schema_version",
        "model_fingerprint",
        "observation_identity",
        "action_spec",
    }
    if set(document) != required:
        raise PolicyConfigurationError(
            "model manifest must contain exactly: "
            + ", ".join(sorted(required))
        )
    if document["schema_version"] != MODEL_MANIFEST_SCHEMA_VERSION:
        raise PolicyConfigurationError(
            "model manifest schema version is unsupported"
        )
    observation = document["observation_identity"]
    if not isinstance(observation, Mapping) or set(observation) != {
        "encoding",
        "encoding_version",
        "scope_fingerprint",
    }:
        raise PolicyConfigurationError(
            "model manifest observation_identity is invalid"
        )
    action = document["action_spec"]
    if not isinstance(action, Mapping) or set(action) != {"count"}:
        raise PolicyConfigurationError("model manifest action_spec is invalid")
    encoding = observation["encoding"]
    encoding_version = observation["encoding_version"]
    scope_fingerprint = observation["scope_fingerprint"]
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (encoding, encoding_version, scope_fingerprint)
    ):
        raise PolicyConfigurationError(
            "model manifest observation identity values must be non-empty strings"
        )
    count = action["count"]
    if type(count) is not int or count <= 0:
        raise PolicyConfigurationError(
            "model manifest action_spec.count must be a positive integer"
        )
    model_fingerprint = document["model_fingerprint"]
    if not isinstance(model_fingerprint, str):
        raise PolicyConfigurationError(
            "model manifest model_fingerprint must be a string"
        )
    return ModelManifest(
        model_fingerprint=_validated_fingerprint(model_fingerprint),
        observation_identity=ObservationIdentity(
            encoding=encoding,
            encoding_version=encoding_version,
            scope_fingerprint=scope_fingerprint,
        ),
        action_spec=DiscreteActionSpec(count=count),
    )


@dataclass(frozen=True, slots=True)
class OracleStats:
    """Unified source/cache counters for one action oracle."""

    policy_queries: int = 0
    cache_hits: int = 0
    table_hits: int = 0
    model_queries: int = 0


@dataclass(frozen=True, slots=True)
class ActionCacheRecord:
    """One immutable canonical-key/action cache checkpoint record."""

    key: Hashable
    action: int


class ActionOracle(Protocol):
    """Search-facing action-source contract.

    ``policy_query_cost`` must be pure and return the exact increase in
    ``stats.policy_queries`` that the next successful ``action`` call for the
    same state will cause.  Cached calls therefore cost zero.
    """

    fingerprint: str

    @property
    def source_description(self) -> Mapping[str, object]: ...

    @property
    def stats(self) -> OracleStats: ...

    def policy_query_cost(self, state: object) -> int: ...

    def action(self, state: object) -> int: ...

    def has_cached(self, state: object) -> bool: ...

    def export_cache(self) -> tuple[ActionCacheRecord, ...]: ...

    def restore_cache(
        self,
        records: Mapping[Hashable, object] | Sequence[ActionCacheRecord],
    ) -> None: ...


def normalize_discrete_action(value: object, action_count: int) -> int:
    """Normalize accepted integer scalar forms and enforce the action space.

    Accepted values are Python/NumPy integer scalars, zero-dimensional integer
    arrays, and integer arrays of shape ``(1,)``.  Booleans, floating-point
    values, strings, nested arrays, and batched arrays are rejected rather than
    coerced.
    """

    if isinstance(action_count, bool) or not isinstance(action_count, Integral):
        raise ActionValidationError(
            "action_count must be a positive Python integer"
        )
    normalized_count = int(action_count)
    if normalized_count <= 0:
        raise ActionValidationError("action_count must be greater than zero")

    scalar: object
    if isinstance(value, np.ndarray):
        if value.shape == ():
            scalar = value.item()
        elif value.shape == (1,):
            scalar = value[0]
        else:
            raise ActionShapeError(
                "discrete action array must have shape () or (1,), "
                f"got {value.shape}"
            )
        if not np.issubdtype(value.dtype, np.integer):
            raise ActionValidationError(
                "discrete action must have an integer dtype, "
                f"got {value.dtype}"
            )
    else:
        scalar = value

    if isinstance(scalar, (bool, np.bool_)) or not isinstance(
        scalar,
        (Integral, np.integer),
    ):
        raise ActionValidationError(
            "discrete action must be an integer scalar, "
            f"got {type(scalar).__name__}"
        )

    action = int(scalar)
    if action < 0 or action >= normalized_count:
        raise ActionValidationError(
            "discrete action is outside the declared range "
            f"[0, {normalized_count}): {action}"
        )
    return action


class _CachedActionOracle:
    """Shared canonical-state cache and counter implementation."""

    def __init__(self, connector: object, *, fingerprint: str) -> None:
        self._connector = connector
        self._action_count = _connector_action_count(connector)
        self.fingerprint = _validated_fingerprint(fingerprint)
        self._cache: dict[Hashable, int] = {}
        self._policy_queries = 0
        self._cache_hits = 0
        self._table_hits = 0
        self._model_queries = 0

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> OracleStats:
        return OracleStats(
            policy_queries=self._policy_queries,
            cache_hits=self._cache_hits,
            table_hits=self._table_hits,
            model_queries=self._model_queries,
        )

    def action(self, state: object) -> int:
        canonical, key = self._canonical_state_and_key(state)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        self._policy_queries += 1
        raw_action = self._query_uncached(canonical)
        action = normalize_discrete_action(raw_action, self._action_count)
        self._cache[key] = action
        return action

    def policy_query_cost(self, state: object) -> int:
        """Return the exact uncached-query cost without mutating the oracle."""

        _, key = self._canonical_state_and_key(state)
        return 0 if key in self._cache else 1

    def has_cached(self, state: object) -> bool:
        _, key = self._canonical_state_and_key(state)
        return key in self._cache

    def export_cache(self) -> tuple[ActionCacheRecord, ...]:
        """Return immutable records without narrowing the Hashable key contract.

        Cache record order has no semantic meaning.  Retaining insertion order
        avoids serializing connector-owned state keys solely to sort them;
        persistence remains the artifact codec's responsibility.
        """

        return tuple(
            ActionCacheRecord(key=key, action=action)
            for key, action in self._cache.items()
        )

    def restore_cache(
        self,
        records: Mapping[Hashable, object] | Sequence[ActionCacheRecord],
    ) -> None:
        """Atomically replace the cache after validating every record."""

        try:
            if isinstance(records, Mapping):
                entries: object = tuple(records.items())
            elif isinstance(records, Sequence) and not isinstance(
                records,
                (str, bytes, bytearray),
            ):
                entries = records
            else:
                raise TypeError("cache must be a mapping or sequence of records")

            restored: dict[Hashable, int] = {}
            for record in entries:  # type: ignore[union-attr]
                if isinstance(record, ActionCacheRecord):
                    key, raw_action = record.key, record.action
                elif isinstance(records, Mapping) and isinstance(record, tuple) and len(record) == 2:
                    key, raw_action = record
                else:
                    raise TypeError(
                        "cache entries must be ActionCacheRecord instances"
                    )
                _require_hashable(key, label="cache key")
                action = normalize_discrete_action(raw_action, self._action_count)
                if key in restored and restored[key] != action:
                    raise ValueError(f"conflicting cache actions for key {key!r}")
                restored[key] = action
        except (ActionValidationError, TypeError, ValueError) as error:
            raise CacheRestoreError(f"invalid action cache: {error}") from error

        self._cache = restored

    def _canonical_state_and_key(self, state: object) -> tuple[object, Hashable]:
        canonicalize = _required_callable(
            self._connector,
            "canonicalize",
            "connector",
        )
        validate_state = _required_callable(
            self._connector,
            "validate_state",
            "connector",
        )
        state_key = _required_callable(self._connector, "state_key", "connector")
        canonical = canonicalize(state)
        validate_state(canonical)
        key = state_key(canonical)
        _require_hashable(key, label="connector state key")
        return canonical, key

    def _query_uncached(self, canonical_state: object) -> object:
        raise NotImplementedError


class TableActionOracle(_CachedActionOracle):
    """Strict precomputed table source with primitive fingerprintable keys."""

    def __init__(
        self,
        connector: object,
        table: Mapping[Hashable, object],
        *,
        source_fingerprint: str | None = None,
    ) -> None:
        action_count = _connector_action_count(connector)
        normalized_table = _validated_table(table, action_count)
        self._content_fingerprint = _table_fingerprint(
            normalized_table,
            action_count=action_count,
            missing_key_policy="error",
        )
        fingerprint, self._declared_fingerprint = _bind_table_fingerprint(
            self._content_fingerprint,
            source_fingerprint,
        )
        super().__init__(connector, fingerprint=fingerprint)
        self._table = normalized_table

    @property
    def source_description(self) -> Mapping[str, object]:
        return {
            "source": "table",
            "fingerprint": self.fingerprint,
            "content_fingerprint": self._content_fingerprint,
            "declared_fingerprint": self._declared_fingerprint,
            "action_count": self._action_count,
            "action_normalization_schema_version": (
                ACTION_NORMALIZATION_SCHEMA_VERSION
            ),
            "missing_key_policy": "error",
        }

    def _query_uncached(self, canonical_state: object) -> object:
        lookup_key = _policy_lookup_key(self._connector, canonical_state)
        try:
            action = self._table[lookup_key]
        except KeyError as error:
            raise UnknownTableKeyError(
                f"precomputed policy table has no key {lookup_key!r}"
            ) from error
        self._table_hits += 1
        return action


class ModelActionOracle(_CachedActionOracle):
    """Deterministic model source validated against connector-owned specs."""

    def __init__(
        self,
        connector: object,
        model: object,
        *,
        source_fingerprint: str,
        manifest: ModelManifest,
    ) -> None:
        self._model = model
        self._model_identity, fingerprint = _validated_model_binding(
            connector,
            source_fingerprint,
            manifest,
        )
        self._observation_shape, self._observation_dtype = (
            _validate_model_compatibility(connector, model)
        )
        super().__init__(connector, fingerprint=fingerprint)

    @property
    def source_description(self) -> Mapping[str, object]:
        return {
            "source": "model",
            "fingerprint": self.fingerprint,
            **self._model_identity,
        }

    def _query_uncached(self, canonical_state: object) -> object:
        observation = _encoded_observation(
            self._connector,
            canonical_state,
            expected_shape=self._observation_shape,
            expected_dtype=self._observation_dtype,
        )
        predict = _required_callable(self._model, "predict", "policy model")
        self._model_queries += 1
        try:
            prediction = predict(observation, deterministic=True)
        except TypeError as error:
            raise ModelCompatibilityError(
                "policy model predict() must accept deterministic=True"
            ) from error
        if not isinstance(prediction, tuple) or len(prediction) != 2:
            raise ModelCompatibilityError(
                "policy model predict() must return an (action, state) pair"
            )
        return prediction[0]


class TableThenModelActionOracle(_CachedActionOracle):
    """Table-first source with primitive keys and deterministic model fallback."""

    def __init__(
        self,
        connector: object,
        table: Mapping[Hashable, object],
        model: object,
        *,
        table_fingerprint: str | None = None,
        model_fingerprint: str,
        model_manifest: ModelManifest,
    ) -> None:
        action_count = _connector_action_count(connector)
        self._table = _validated_table(table, action_count)
        self._table_content_fingerprint = _table_fingerprint(
            self._table,
            action_count=action_count,
            missing_key_policy="model_fallback",
        )
        (
            self._table_fingerprint,
            self._declared_table_fingerprint,
        ) = _bind_table_fingerprint(
            self._table_content_fingerprint,
            table_fingerprint,
        )
        self._model_identity, self._model_policy_fingerprint = (
            _validated_model_binding(
                connector,
                model_fingerprint,
                model_manifest,
            )
        )
        self._model_fingerprint = str(
            self._model_identity["model_fingerprint"]
        )
        self._model = model
        self._observation_shape, self._observation_dtype = (
            _validate_model_compatibility(connector, model)
        )
        combined = _hash_json(
            {
                "source": "table_then_model",
                "table_fingerprint": self._table_fingerprint,
                "model_policy_fingerprint": self._model_policy_fingerprint,
            }
        )
        super().__init__(connector, fingerprint=combined)

    @property
    def source_description(self) -> Mapping[str, object]:
        return {
            "source": "table_then_model",
            "fingerprint": self.fingerprint,
            "table_fingerprint": self._table_fingerprint,
            "table_content_fingerprint": self._table_content_fingerprint,
            "declared_table_fingerprint": self._declared_table_fingerprint,
            "model_fingerprint": self._model_fingerprint,
            "model_policy_fingerprint": self._model_policy_fingerprint,
            "model_manifest": self._model_identity["model_manifest"],
            "action_count": self._action_count,
            "action_normalization_schema_version": (
                ACTION_NORMALIZATION_SCHEMA_VERSION
            ),
            "missing_key_policy": "model_fallback",
        }

    def _query_uncached(self, canonical_state: object) -> object:
        lookup_key = _policy_lookup_key(self._connector, canonical_state)
        if lookup_key in self._table:
            self._table_hits += 1
            return self._table[lookup_key]

        observation = _encoded_observation(
            self._connector,
            canonical_state,
            expected_shape=self._observation_shape,
            expected_dtype=self._observation_dtype,
        )
        predict = _required_callable(self._model, "predict", "policy model")
        self._model_queries += 1
        try:
            prediction = predict(observation, deterministic=True)
        except TypeError as error:
            raise ModelCompatibilityError(
                "policy model predict() must accept deterministic=True"
            ) from error
        if not isinstance(prediction, tuple) or len(prediction) != 2:
            raise ModelCompatibilityError(
                "policy model predict() must return an (action, state) pair"
            )
        return prediction[0]


def _connector_action_count(connector: object) -> int:
    try:
        action_spec = getattr(connector, "action_spec")
        count = getattr(action_spec, "count")
    except AttributeError as error:
        raise PolicyConfigurationError(
            "connector must declare action_spec.count"
        ) from error
    # Reuse the normalization validator without accepting booleans/floats.
    if isinstance(count, bool) or not isinstance(count, Integral) or int(count) <= 0:
        raise PolicyConfigurationError(
            "connector action_spec.count must be a positive integer"
        )
    return int(count)


def _validated_table(
    table: Mapping[Hashable, object],
    action_count: int,
) -> dict[Hashable, int]:
    if not isinstance(table, Mapping):
        raise PolicyConfigurationError("policy table must be a mapping")
    normalized: dict[Hashable, int] = {}
    for key, raw_action in table.items():
        _require_hashable(key, label="policy table key")
        try:
            normalized[key] = normalize_discrete_action(raw_action, action_count)
        except (ActionShapeError, ActionValidationError) as error:
            raise type(error)(
                f"invalid action for policy table key {key!r}: {error}"
            ) from error
    return normalized


def _policy_lookup_key(connector: object, canonical_state: object) -> Hashable:
    lookup = _required_callable(connector, "policy_lookup_key", "connector")
    key = lookup(canonical_state)
    _require_hashable(key, label="policy lookup key")
    return key


def _validate_model_compatibility(
    connector: object,
    model: object,
) -> tuple[tuple[int, ...], np.dtype[object]]:
    try:
        observation_spec = getattr(connector, "observation_spec")
        expected_shape = tuple(getattr(observation_spec, "shape"))
        expected_dtype = np.dtype(getattr(observation_spec, "dtype"))
    except (AttributeError, TypeError, ValueError) as error:
        raise PolicyConfigurationError(
            "connector must declare a valid observation_spec.shape and dtype"
        ) from error

    try:
        observation_space = getattr(model, "observation_space")
        model_shape = tuple(getattr(observation_space, "shape"))
        model_dtype = np.dtype(getattr(observation_space, "dtype"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ModelCompatibilityError(
            "policy model must declare observation space shape and dtype"
        ) from error

    if model_shape != expected_shape:
        raise ModelCompatibilityError(
            "policy model observation space shape is incompatible with the "
            f"connector: expected {expected_shape}, got {model_shape}"
        )
    if model_dtype != expected_dtype:
        raise ModelCompatibilityError(
            "policy model observation space dtype is incompatible with the "
            f"connector: expected {expected_dtype}, got {model_dtype}"
        )

    expected_actions = _connector_action_count(connector)
    try:
        action_space = getattr(model, "action_space")
        model_actions = getattr(action_space, "n")
    except AttributeError as error:
        raise ModelCompatibilityError(
            "policy model must declare a scalar discrete action space"
        ) from error
    if (
        isinstance(model_actions, bool)
        or not isinstance(model_actions, Integral)
        or int(model_actions) != expected_actions
    ):
        raise ModelCompatibilityError(
            "policy model action space is incompatible with the connector: "
            f"expected Discrete({expected_actions}), got n={model_actions!r}"
        )
    _required_callable(model, "predict", "policy model")
    return expected_shape, expected_dtype


def _validated_model_binding(
    connector: object,
    source_fingerprint: str,
    manifest: ModelManifest,
) -> tuple[dict[str, object], str]:
    """Validate a model-owned manifest and derive its policy identity."""

    fingerprint = _validated_fingerprint(source_fingerprint)
    if not isinstance(manifest, ModelManifest):
        raise ModelCompatibilityError(
            "policy model requires a versioned ModelManifest"
        )
    if manifest.schema_version != MODEL_MANIFEST_SCHEMA_VERSION:
        raise ModelCompatibilityError(
            "policy model manifest schema version is unsupported"
        )
    manifest_fingerprint = _validated_fingerprint(manifest.model_fingerprint)
    if manifest_fingerprint != fingerprint:
        raise ModelCompatibilityError(
            "policy model fingerprint does not match its manifest"
        )

    try:
        observation_spec = getattr(connector, "observation_spec")
        expected_observation_identity = getattr(observation_spec, "identity")
    except AttributeError as error:
        raise PolicyConfigurationError(
            "connector must declare observation_spec.identity"
        ) from error
    if not isinstance(expected_observation_identity, ObservationIdentity):
        raise PolicyConfigurationError(
            "connector observation_spec.identity must be ObservationIdentity"
        )
    if manifest.observation_identity != expected_observation_identity:
        raise ModelCompatibilityError(
            "policy model observation identity is incompatible with the connector"
        )

    expected_action_count = _connector_action_count(connector)
    if (
        not isinstance(manifest.action_spec, DiscreteActionSpec)
        or manifest.action_spec.count != expected_action_count
    ):
        raise ModelCompatibilityError(
            "policy model action contract is incompatible with the connector"
        )

    manifest_document = asdict(manifest)
    identity: dict[str, object] = {
        "model_fingerprint": fingerprint,
        "model_manifest": manifest_document,
        "deterministic": True,
        "action_normalization_schema_version": (
            ACTION_NORMALIZATION_SCHEMA_VERSION
        ),
    }
    policy_fingerprint = _model_policy_fingerprint(
        model_fingerprint=fingerprint,
        model_manifest=manifest_document,
        deterministic=True,
        action_normalization_schema_version=(
            ACTION_NORMALIZATION_SCHEMA_VERSION
        ),
    )
    return identity, policy_fingerprint


def _model_policy_fingerprint(
    *,
    model_fingerprint: str,
    model_manifest: Mapping[str, object],
    deterministic: bool,
    action_normalization_schema_version: int,
) -> str:
    return _hash_json(
        {
            "schema": "stache.model-policy-binding/v1",
            "model_fingerprint": model_fingerprint,
            "model_manifest": dict(model_manifest),
            "deterministic": deterministic,
            "action_normalization_schema_version": (
                action_normalization_schema_version
            ),
        }
    )


def custom_policy_source(
    identity: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    """Build a canonically bound source descriptor for a custom oracle."""

    if not isinstance(identity, Mapping) or any(
        type(key) is not str for key in identity
    ):
        raise PolicyConfigurationError(
            "custom policy identity must be a string-keyed mapping"
        )
    material = dict(identity)
    fingerprint = _hash_json(
        {"schema": "stache.custom-policy-binding/v1", "identity": material}
    )
    return fingerprint, {
        "source": "custom",
        "fingerprint": fingerprint,
        "identity": material,
    }


def policy_fingerprint_from_source(source: Mapping[str, object]) -> str:
    """Recompute and validate the canonical fingerprint of a source descriptor."""

    if not isinstance(source, Mapping) or any(
        type(key) is not str for key in source
    ):
        raise PolicyConfigurationError(
            "policy source must be a string-keyed mapping"
        )
    kind = source.get("source")
    if kind == "table":
        required = {
            "source",
            "fingerprint",
            "content_fingerprint",
            "declared_fingerprint",
            "action_count",
            "action_normalization_schema_version",
            "missing_key_policy",
        }
        _require_source_fields(source, required)
        if source["missing_key_policy"] != "error":
            raise PolicyConfigurationError(
                "table policy source has an invalid missing-key policy"
            )
        _validate_source_action_contract(source)
        content = _validated_fingerprint(str(source["content_fingerprint"]))
        declared_value = source["declared_fingerprint"]
        if declared_value is not None and not isinstance(declared_value, str):
            raise PolicyConfigurationError(
                "table declared_fingerprint must be a string or null"
            )
        derived, _ = _bind_table_fingerprint(content, declared_value)
    elif kind == "model":
        required = {
            "source",
            "fingerprint",
            "model_fingerprint",
            "model_manifest",
            "deterministic",
            "action_normalization_schema_version",
        }
        _require_source_fields(source, required)
        manifest = source["model_manifest"]
        if not isinstance(manifest, Mapping):
            raise PolicyConfigurationError("model_manifest must be a mapping")
        parsed = model_manifest_from_document(manifest)
        model_fingerprint = _validated_fingerprint(
            str(source["model_fingerprint"])
        )
        if parsed.model_fingerprint != model_fingerprint:
            raise PolicyConfigurationError(
                "model source fingerprint disagrees with its manifest"
            )
        if source["deterministic"] is not True:
            raise PolicyConfigurationError(
                "model source must declare deterministic prediction"
            )
        normalization = source["action_normalization_schema_version"]
        if normalization != ACTION_NORMALIZATION_SCHEMA_VERSION:
            raise PolicyConfigurationError(
                "model source action normalization version is unsupported"
            )
        derived = _model_policy_fingerprint(
            model_fingerprint=model_fingerprint,
            model_manifest=dict(manifest),
            deterministic=True,
            action_normalization_schema_version=normalization,
        )
    elif kind == "table_then_model":
        required = {
            "source",
            "fingerprint",
            "table_fingerprint",
            "table_content_fingerprint",
            "declared_table_fingerprint",
            "model_fingerprint",
            "model_policy_fingerprint",
            "model_manifest",
            "action_count",
            "action_normalization_schema_version",
            "missing_key_policy",
        }
        _require_source_fields(source, required)
        if source["missing_key_policy"] != "model_fallback":
            raise PolicyConfigurationError(
                "table-then-model source has an invalid missing-key policy"
            )
        _validate_source_action_contract(source)
        table_content = _validated_fingerprint(
            str(source["table_content_fingerprint"])
        )
        declared = source["declared_table_fingerprint"]
        if declared is not None and not isinstance(declared, str):
            raise PolicyConfigurationError(
                "declared_table_fingerprint must be a string or null"
            )
        table_fingerprint, _ = _bind_table_fingerprint(table_content, declared)
        if table_fingerprint != source["table_fingerprint"]:
            raise PolicyConfigurationError(
                "table source fingerprint is internally inconsistent"
            )
        manifest = source["model_manifest"]
        if not isinstance(manifest, Mapping):
            raise PolicyConfigurationError("model_manifest must be a mapping")
        parsed = model_manifest_from_document(manifest)
        model_fingerprint = _validated_fingerprint(
            str(source["model_fingerprint"])
        )
        if parsed.model_fingerprint != model_fingerprint:
            raise PolicyConfigurationError(
                "model source fingerprint disagrees with its manifest"
            )
        model_policy_fingerprint = _model_policy_fingerprint(
            model_fingerprint=model_fingerprint,
            model_manifest=dict(manifest),
            deterministic=True,
            action_normalization_schema_version=(
                ACTION_NORMALIZATION_SCHEMA_VERSION
            ),
        )
        if model_policy_fingerprint != source["model_policy_fingerprint"]:
            raise PolicyConfigurationError(
                "model policy fingerprint is internally inconsistent"
            )
        derived = _hash_json(
            {
                "source": "table_then_model",
                "table_fingerprint": table_fingerprint,
                "model_policy_fingerprint": model_policy_fingerprint,
            }
        )
    elif kind == "custom":
        required = {"source", "fingerprint", "identity"}
        _require_source_fields(source, required)
        identity = source["identity"]
        if not isinstance(identity, Mapping):
            raise PolicyConfigurationError(
                "custom policy identity must be a mapping"
            )
        derived, _ = custom_policy_source(identity)
    else:
        raise PolicyConfigurationError(f"unknown policy source kind: {kind!r}")

    declared_fingerprint = source.get("fingerprint")
    if declared_fingerprint != derived:
        raise PolicyConfigurationError(
            "policy source fingerprint is internally inconsistent"
        )
    return derived


def policy_fingerprint_for_connector(
    source: Mapping[str, object],
    connector: object,
) -> str:
    """Validate persisted policy semantics against a requested connector."""

    fingerprint = policy_fingerprint_from_source(source)
    expected_action_count = _connector_action_count(connector)
    kind = source["source"]
    if kind in {"table", "table_then_model"}:
        if source["action_count"] != expected_action_count:
            raise ModelCompatibilityError(
                "persisted policy action contract is incompatible with the connector"
            )
    if kind in {"model", "table_then_model"}:
        manifest_document = source["model_manifest"]
        if not isinstance(manifest_document, Mapping):
            raise ModelCompatibilityError(
                "persisted policy model manifest is invalid"
            )
        manifest = model_manifest_from_document(manifest_document)
        try:
            observation_identity = connector.observation_spec.identity
        except AttributeError as error:
            raise PolicyConfigurationError(
                "connector must declare observation_spec.identity"
            ) from error
        if manifest.observation_identity != observation_identity:
            raise ModelCompatibilityError(
                "persisted policy observation identity is incompatible with "
                "the connector"
            )
        if manifest.action_spec.count != expected_action_count:
            raise ModelCompatibilityError(
                "persisted policy action contract is incompatible with the connector"
            )
    return fingerprint


def _require_source_fields(
    source: Mapping[str, object],
    required: set[str],
) -> None:
    if set(source) != required:
        raise PolicyConfigurationError(
            "policy source fields do not match its declared source kind"
        )


def _validate_source_action_contract(source: Mapping[str, object]) -> None:
    count = source["action_count"]
    if type(count) is not int or count <= 0:
        raise PolicyConfigurationError(
            "policy source action_count must be a positive integer"
        )
    if (
        source["action_normalization_schema_version"]
        != ACTION_NORMALIZATION_SCHEMA_VERSION
    ):
        raise PolicyConfigurationError(
            "policy source action normalization version is unsupported"
        )


def _encoded_observation(
    connector: object,
    canonical_state: object,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[object],
) -> np.ndarray:
    encode = _required_callable(connector, "encode_observation", "connector")
    observation = np.asarray(encode(canonical_state))
    if observation.shape != expected_shape:
        raise ModelCompatibilityError(
            "connector encoded observation shape violates observation_spec: "
            f"expected {expected_shape}, got {observation.shape}"
        )
    if observation.dtype != expected_dtype:
        raise ModelCompatibilityError(
            "connector encoded observation dtype violates observation_spec: "
            f"expected {expected_dtype}, got {observation.dtype}"
        )
    return np.array(observation, copy=True)


def _required_callable(owner: object, name: str, label: str):
    try:
        value = getattr(owner, name)
    except AttributeError as error:
        raise PolicyConfigurationError(
            f"{label} must provide callable {name}()"
        ) from error
    if not callable(value):
        raise PolicyConfigurationError(f"{label}.{name} must be callable")
    return value


def _validated_fingerprint(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyConfigurationError("source fingerprint must be a non-empty string")
    return value


def _table_fingerprint(
    table: Mapping[Hashable, int],
    *,
    action_count: int,
    missing_key_policy: str,
) -> str:
    try:
        entries = sorted(
            (
                {"key": _stable_node(key), "action": action}
                for key, action in table.items()
            ),
            key=lambda entry: json.dumps(
                entry["key"],
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    except (TypeError, ValueError) as error:
        raise PolicyConfigurationError(
            "policy table fingerprint keys must use finite primitive scalars "
            f"or tuples: {error}"
        ) from error
    return _hash_json(
        {
            "schema": "stache.policy-table-fingerprint/v1",
            "action_count": action_count,
            "action_normalization_schema_version": (
                ACTION_NORMALIZATION_SCHEMA_VERSION
            ),
            "missing_key_policy": missing_key_policy,
            "entries": entries,
        }
    )


def _bind_table_fingerprint(
    content_fingerprint: str,
    declared_fingerprint: str | None,
) -> tuple[str, str | None]:
    if declared_fingerprint is None:
        return content_fingerprint, None
    declared = _validated_fingerprint(declared_fingerprint)
    return (
        _hash_json(
            {
                "schema": "stache.policy-table-binding/v1",
                "content_fingerprint": content_fingerprint,
                "declared_fingerprint": declared,
            }
        ),
        declared,
    )


def _stable_node(value: object) -> object:
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, (bool, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, (Integral, np.integer)):
        return {"type": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite floating-point keys are not supported")
        return {"type": "float", "value": number}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [_stable_node(item) for item in value]}
    raise TypeError(
        "policy table keys must use primitive scalars or tuples, "
        f"got {type(value).__name__}"
    )


def _hash_json(value: object) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("identity material must be a mapping")
    try:
        return fingerprint_document(value)
    except IdentityEncodingError as error:
        raise PolicyConfigurationError(str(error)) from error


def _require_hashable(value: object, *, label: str) -> None:
    try:
        hash(value)
    except TypeError as error:
        raise PolicyConfigurationError(
            f"{label} must be hashable, got {type(value).__name__}"
        ) from error
