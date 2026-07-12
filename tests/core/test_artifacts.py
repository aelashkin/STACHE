"""Primitive, versioned RR result artifact contracts.

These tests use the domain-neutral toy graph instead of Taxi, Gymnasium, or an
SB3 checkpoint.  The artifact layer must delegate state/key meaning to the
connector and must never rely on Python-specific YAML constructors.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from stache.explainability import artifacts
from stache.explainability.artifacts import (
    ARTIFACT_SCHEMA,
    ARTIFACT_VERSION,
    ArtifactCompatibilityError,
    ArtifactError,
    ArtifactSchemaError,
    document_to_result,
    load_result,
    result_to_document,
    save_result,
)
from stache.explainability.core.models import (
    CounterfactualSelection,
    MinimumBasis,
    SearchExtent,
    SearchOptions,
)
from stache.explainability.core.search import compute_rr

from ._toy import (
    ToyConnector,
    ToyOracle,
    disconnected_formal_minimum_space,
    exact_space,
    query_budget_space,
    tied_minimum_space,
)


JsonPrimitive = None | bool | int | float | str
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


class ArtifactToyConnector(ToyConnector):
    """Toy connector with a strict, independently versioned artifact codec."""

    def __init__(self, space=None) -> None:
        super().__init__(space or exact_space())
        self.identity = replace(
            self.identity,
            codec="toy-state-record",
            codec_version="1",
        )

    @property
    def artifact_codec(self) -> "ArtifactToyConnector":
        return self

    def encode_state(self, state: str) -> JsonValue:
        canonical = self.canonicalize(state)
        self.validate_state(canonical)
        return {"name": canonical}

    def decode_state(self, value: JsonValue) -> str:
        if (
            not isinstance(value, dict)
            or set(value) != {"name"}
            or not isinstance(value["name"], str)
        ):
            raise ValueError("toy artifact state must be {'name': <str>}")
        state = self.canonicalize(value["name"])
        self.validate_state(state)
        return state

    def encode_key(self, key: str) -> JsonValue:
        if not isinstance(key, str):
            raise TypeError("toy artifact key must be a string")
        return {"canonical": key}

    def decode_key(self, value: JsonValue) -> str:
        if (
            not isinstance(value, dict)
            or set(value) != {"canonical"}
            or not isinstance(value["canonical"], str)
        ):
            raise ValueError("toy artifact key must be {'canonical': <str>}")
        return value["canonical"]


class LossyStateConnector(ArtifactToyConnector):
    def decode_state(self, value: JsonValue) -> str:
        super().decode_state(value)
        return "a"


class LossyKeyConnector(ArtifactToyConnector):
    def decode_key(self, value: JsonValue) -> str:
        return f"{super().decode_key(value)}-changed"


class NonPrimitiveCodecConnector(ArtifactToyConnector):
    def encode_state(self, state: str) -> Any:
        return (state,)  # A tuple is intentionally outside the external schema.


def exact_result(connector: ArtifactToyConnector | None = None) -> tuple[object, ArtifactToyConnector]:
    connector = connector or ArtifactToyConnector()
    result = compute_rr(
        "s",
        connector,
        ToyOracle(connector.space.actions, fingerprint="toy-policy-sha256"),
        SearchOptions(
            counterfactuals=CounterfactualSelection.BOTH,
            minimum_basis=MinimumBasis.GRAPH_BOUNDARY,
            extent=SearchExtent.EXACT,
        ),
    )
    return result, connector


def assert_recursively_primitive(value: object, *, path: str = "document") -> None:
    """Assert the interchange document contains only JSON/YAML-safe values."""

    if value is None or type(value) in {bool, int, float, str}:
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_recursively_primitive(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert type(key) is str, f"{path} contains a non-string key: {key!r}"
            assert_recursively_primitive(item, path=f"{path}.{key}")
        return
    pytest.fail(f"{path} contains non-primitive {type(value).__name__}: {value!r}")


def make_document() -> tuple[dict[str, object], object, ArtifactToyConnector]:
    result, connector = exact_result()
    provenance: Mapping[str, object] = {
        "git": {"commit": "abc123", "dirty": False},
        "dependencies": {"python": "3.11.15", "pytest": "9.1.1"},
    }
    document = result_to_document(
        result,
        connector,
        provenance=provenance,
    )
    return document, result, connector


def test_result_document_is_versioned_complete_and_recursively_primitive() -> None:
    document, result, connector = make_document()

    assert document["schema"] == ARTIFACT_SCHEMA == "stache.rr-result"
    assert document["schema_version"] == ARTIFACT_VERSION == 2
    assert document["connector"] == {
        "domain": connector.identity.domain,
        "connector_version": connector.identity.connector_version,
        "state_universe": connector.identity.state_universe,
        "state_universe_version": connector.identity.state_universe_version,
        "metric": connector.identity.metric,
        "metric_version": connector.identity.metric_version,
        "object_projection": connector.identity.object_projection,
        "object_projection_version": connector.identity.object_projection_version,
        "factorization": connector.identity.factorization,
        "factorization_version": connector.identity.factorization_version,
        "topology": connector.identity.topology,
        "topology_version": connector.identity.topology_version,
        "adjacency_threshold": connector.identity.adjacency_threshold,
        "codec": connector.identity.codec,
        "codec_version": connector.identity.codec_version,
    }
    assert document["metric_certificate"]["scope_fingerprint"] == (
        connector.metric_certificate.scope_fingerprint
    )
    assert document["policy"]["fingerprint"] == result.metadata.policy_fingerprint
    assert document["policy"]["action_normalization_schema_version"] == 1
    assert document["options"]["minimum_basis"] == "graph_boundary"
    assert document["result"]["counterfactual_existence"] == "found"
    assert document["result"]["completeness"]["region_complete"] is True
    assert document["result"]["stop_reason"] == "completed"
    assert document["provenance"]["git"]["commit"] == "abc123"

    assert_recursively_primitive(document)
    dumped = yaml.safe_dump(document, sort_keys=True)
    assert "!!python" not in dumped
    assert yaml.safe_load(dumped) == document


def test_document_round_trip_restores_the_typed_immutable_result() -> None:
    document, expected, connector = make_document()

    restored = document_to_result(
        yaml.safe_load(yaml.safe_dump(document)),
        connector,
        expected_policy_fingerprint=expected.metadata.policy_fingerprint,
    )

    assert restored == expected
    assert isinstance(restored.region, tuple)
    assert isinstance(restored.boundary_counterfactuals, tuple)
    assert isinstance(restored.minimal_counterfactuals, tuple)


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        pytest.param("schema", "other.schema", id="schema-name"),
        pytest.param("schema_version", 999, id="schema-version"),
    ],
)
def test_document_loader_rejects_unknown_schema_or_version(
    field: str,
    invalid_value: object,
) -> None:
    document, _, connector = make_document()
    document[field] = invalid_value

    with pytest.raises(ArtifactSchemaError, match=field.replace("_", "[ _]")):
        document_to_result(document, connector)


def test_document_loader_rejects_unknown_core_schema_version() -> None:
    document, _, connector = make_document()
    document["metadata"]["core_schema_version"] = 999

    with pytest.raises(ArtifactSchemaError, match="core.schema.version"):
        document_to_result(document, connector)


def test_document_loader_rejects_connector_identity_mismatch() -> None:
    document, _, connector = make_document()
    connector.identity = replace(connector.identity, connector_version="2")

    with pytest.raises(ArtifactCompatibilityError, match="connector_version"):
        document_to_result(document, connector)


def test_document_loader_rejects_expected_policy_fingerprint_mismatch() -> None:
    document, _, connector = make_document()

    with pytest.raises(ArtifactCompatibilityError, match="policy.*fingerprint"):
        document_to_result(
            document,
            connector,
            expected_policy_fingerprint="different-policy-sha256",
        )


def test_document_loader_rejects_internal_policy_fingerprint_contradiction() -> None:
    document, _, connector = make_document()
    document["policy"]["fingerprint"] = "sha256:" + "0" * 64

    with pytest.raises(ArtifactCompatibilityError, match="policy.*fingerprint"):
        document_to_result(document, connector)


def test_document_loader_rejects_policy_source_fingerprint_contradiction() -> None:
    document, _, connector = make_document()
    document["policy"]["source"]["fingerprint"] = "sha256:" + "1" * 64

    with pytest.raises(ArtifactCompatibilityError, match="policy.*source"):
        document_to_result(document, connector)


def test_document_loader_rejects_internal_search_fingerprint_contradiction() -> None:
    document, _, connector = make_document()
    document["metadata"]["search_fingerprint"] = "sha256:" + "2" * 64

    with pytest.raises(ArtifactCompatibilityError, match="search.*fingerprint"):
        document_to_result(document, connector)


def test_document_loader_rejects_action_normalization_version_mismatch() -> None:
    document, _, connector = make_document()
    document["policy"]["action_normalization_schema_version"] = 999

    with pytest.raises(
        ArtifactCompatibilityError,
        match="action.normalization.*version",
    ):
        document_to_result(document, connector)


@pytest.mark.parametrize(
    "connector_type, diagnostic",
    [
        pytest.param(LossyStateConnector, "state.*round.trip", id="state"),
        pytest.param(LossyKeyConnector, "key.*round.trip", id="key"),
        pytest.param(NonPrimitiveCodecConnector, "primitive", id="primitive"),
    ],
)
def test_writer_rejects_lossy_or_nonprimitive_connector_codecs(
    connector_type: type[ArtifactToyConnector],
    diagnostic: str,
) -> None:
    result, _ = exact_result()
    connector = connector_type()

    with pytest.raises(ArtifactError, match=diagnostic):
        result_to_document(result, connector)


def test_loader_rejects_encoded_state_key_disagreement() -> None:
    document, _, connector = make_document()
    document["result"]["seed"]["key"] = {"canonical": "a"}

    with pytest.raises(ArtifactSchemaError, match="seed.*key|key.*seed"):
        document_to_result(document, connector)


def test_yaml_save_load_round_trip_and_refuse_overwrite(tmp_path: Path) -> None:
    result, connector = exact_result()
    target = tmp_path / "result.yaml"

    save_result(target, result, connector, provenance={"git": {"commit": "abc123"}})
    original = target.read_bytes()
    restored = load_result(
        target,
        connector,
        expected_policy_fingerprint=result.metadata.policy_fingerprint,
    )

    assert restored == result
    assert b"!!python" not in original
    with pytest.raises(ArtifactError, match="exist|overwrite"):
        save_result(target, result, connector)
    assert target.read_bytes() == original


def test_yaml_save_can_atomically_replace_only_when_explicit(tmp_path: Path) -> None:
    result, connector = exact_result()
    target = tmp_path / "result.yaml"
    save_result(target, result, connector, provenance={"revision": 1})

    save_result(
        target,
        result,
        connector,
        provenance={"revision": 2},
        overwrite=True,
    )

    assert yaml.safe_load(target.read_text(encoding="utf-8"))["provenance"] == {
        "revision": 2
    }


def test_budget_continuation_is_a_truthful_nonresumable_summary() -> None:
    connector = ArtifactToyConnector()
    partial = compute_rr(
        "s",
        connector,
        ToyOracle(connector.space.actions, fingerprint="toy-policy-sha256"),
        SearchOptions(max_expanded=0),
    )
    assert partial.continuation is not None

    document = result_to_document(partial, connector)
    summary = document["result"]["continuation"]

    assert summary == {
        "resumable": False,
        "checkpoint_version": partial.continuation.checkpoint_version,
        "fingerprint": partial.continuation.fingerprint,
        "payload_digest": partial.continuation.payload_digest,
        "remaining_frontier_size": partial.completeness.remaining_frontier_size,
    }
    restored = document_to_result(document, connector)
    assert restored.continuation is None
    assert restored.completeness == partial.completeness


def test_prewarmed_seed_with_zero_query_budget_round_trips_truthfully() -> None:
    connector = ArtifactToyConnector()
    oracle = ToyOracle(connector.space.actions)
    oracle.action("s")
    partial = compute_rr(
        "s",
        connector,
        oracle,
        SearchOptions(max_policy_queries=0),
    )

    assert partial.stats.policy_queries == 0
    document = result_to_document(partial, connector)
    restored = document_to_result(document, connector)

    assert restored.stats.policy_queries == 0
    assert restored.completeness == partial.completeness


def test_formal_global_disconnected_minimum_round_trips() -> None:
    connector = ArtifactToyConnector(disconnected_formal_minimum_space())
    result = compute_rr(
        "s",
        connector,
        ToyOracle(connector.space.actions, fingerprint="toy-policy-sha256"),
        SearchOptions(minimum_basis=MinimumBasis.FORMAL_GLOBAL),
    )

    document = result_to_document(result, connector)
    restored = document_to_result(document, connector)

    assert restored == result
    assert restored.minimal_counterfactuals[0].key == "q"
    assert restored.minimal_counterfactuals[0].graph_depth is None


def test_certified_formal_partial_minimum_round_trips_truthfully() -> None:
    connector = ArtifactToyConnector(query_budget_space())
    result = compute_rr(
        "s",
        connector,
        ToyOracle(connector.space.actions, fingerprint="toy-policy-sha256"),
        SearchOptions(
            minimum_basis=MinimumBasis.FORMAL_GLOBAL,
            max_policy_queries=2,
        ),
    )

    restored = document_to_result(result_to_document(result, connector), connector)

    assert restored == replace(result, continuation=None)
    assert restored.robustness_radius == 1.0
    assert restored.completeness.radius_complete
    assert not restored.completeness.minimal_counterfactuals_complete


def test_loader_rejects_duplicate_record_keys_and_missing_seed_membership() -> None:
    duplicate, _, connector = make_document()
    duplicate["result"]["region"].append(deepcopy(duplicate["result"]["region"][0]))
    with pytest.raises(ArtifactSchemaError, match="duplicate.*region|region.*duplicate"):
        document_to_result(duplicate, connector)

    missing_seed, _, connector = make_document()
    seed_key = missing_seed["result"]["seed"]["key"]
    missing_seed["result"]["region"] = [
        item
        for item in missing_seed["result"]["region"]
        if item["key"] != seed_key
    ]
    with pytest.raises(ArtifactSchemaError, match="seed.*region|region.*seed"):
        document_to_result(missing_seed, connector)


def test_loader_recomputes_every_record_formal_distance() -> None:
    document, _, connector = make_document()
    document["result"]["region"][1]["formal_distance"] = 999

    with pytest.raises(
        ArtifactSchemaError,
        match="formal.distance.*connector|connector.*formal.distance",
    ):
        document_to_result(document, connector)


def test_loader_rejects_conflicting_representations_of_one_state_key() -> None:
    document, _, connector = make_document()
    minimum = document["result"]["minimal_counterfactuals"][0]
    minimum["action"] = 2
    document["result"]["counterfactuals"]["minimal"][0]["action"] = 2

    with pytest.raises(ArtifactSchemaError, match="conflicting.*record|record.*conflict"):
        document_to_result(document, connector)


def test_graph_basis_minima_must_be_members_of_the_boundary() -> None:
    document, _, connector = make_document()
    minimum_key = document["result"]["minimal_counterfactuals"][0]["key"]
    boundary = [
        record
        for record in document["result"]["boundary_counterfactuals"]
        if record["key"] != minimum_key
    ]
    document["result"]["boundary_counterfactuals"] = boundary
    document["result"]["counterfactuals"]["boundary"] = deepcopy(boundary)

    with pytest.raises(ArtifactSchemaError, match="minima.*boundary|boundary.*minima"):
        document_to_result(document, connector)


def test_complete_graph_ties_cannot_omit_a_boundary_minimum() -> None:
    connector = ArtifactToyConnector(tied_minimum_space())
    result = compute_rr(
        "s",
        connector,
        ToyOracle(connector.space.actions, fingerprint="toy-policy-sha256"),
        SearchOptions(
            counterfactuals=CounterfactualSelection.BOTH,
            minimum_basis=MinimumBasis.GRAPH_BOUNDARY,
            extent=SearchExtent.EXACT,
        ),
    )
    document = result_to_document(result, connector)
    assert {
        record["key"]["canonical"]
        for record in document["result"]["minimal_counterfactuals"]
    } == {"x", "y"}
    retained = document["result"]["minimal_counterfactuals"][:1]
    document["result"]["minimal_counterfactuals"] = retained
    document["result"]["counterfactuals"]["minimal"] = deepcopy(retained)

    with pytest.raises(ArtifactSchemaError, match="tied|minima.*complete|complete.*minima"):
        document_to_result(document, connector)


def test_complete_graph_result_rejects_false_depth_and_radius_claims() -> None:
    connector = ArtifactToyConnector(tied_minimum_space())
    result = compute_rr(
        "s",
        connector,
        ToyOracle(connector.space.actions, fingerprint="toy-policy-sha256"),
        SearchOptions(),
    )
    document = result_to_document(result, connector)
    for field in (
        "boundary_counterfactuals",
        "minimal_counterfactuals",
    ):
        for record in document["result"][field]:
            if record["key"]["canonical"] in {"x", "y"}:
                record["graph_depth"] = 5
    for field in ("boundary", "minimal"):
        for record in document["result"]["counterfactuals"][field]:
            if record["key"]["canonical"] in {"x", "y"}:
                record["graph_depth"] = 5
    document["result"]["robustness_radius"] = 5
    document["result"]["best_known_radius"] = 5

    with pytest.raises(ArtifactSchemaError, match="depth|radius|BFS"):
        document_to_result(document, connector)


def test_complete_graph_result_rejects_an_omitted_boundary_tie() -> None:
    connector = ArtifactToyConnector(tied_minimum_space())
    result = compute_rr(
        "s",
        connector,
        ToyOracle(connector.space.actions, fingerprint="toy-policy-sha256"),
        SearchOptions(),
    )
    document = result_to_document(result, connector)

    def without_y(records: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            record
            for record in records
            if record["key"]["canonical"] != "y"
        ]

    for field in ("boundary_counterfactuals", "minimal_counterfactuals"):
        document["result"][field] = without_y(document["result"][field])
    for field in ("boundary", "minimal"):
        document["result"]["counterfactuals"][field] = without_y(
            document["result"]["counterfactuals"][field]
        )

    with pytest.raises(ArtifactSchemaError, match="complete|neighbor|boundary"):
        document_to_result(document, connector)


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        pytest.param("discovery_source", "guessed", "discovery.source", id="source"),
        pytest.param("graph_depth", None, "graph.*depth", id="graph-depth"),
    ],
)
def test_loader_rejects_impossible_graph_discovery_metadata(
    field: str,
    value: object,
    diagnostic: str,
) -> None:
    document, _, connector = make_document()
    document["result"]["region"][1][field] = value

    with pytest.raises(ArtifactSchemaError, match=diagnostic):
        document_to_result(document, connector)


def test_loader_requires_seed_origin_metadata_to_be_zero_depth() -> None:
    document, _, connector = make_document()
    seed_key = document["result"]["seed"]["key"]
    document["result"]["seed"]["graph_depth"] = 1
    for record in document["result"]["region"]:
        if record["key"] == seed_key:
            record["graph_depth"] = 1
            break

    with pytest.raises(ArtifactSchemaError, match="seed.*graph.*depth"):
        document_to_result(document, connector)


def test_loader_rejects_action_range_completeness_and_result_invariants() -> None:
    invalid_action, _, connector = make_document()
    invalid_action["result"]["seed"]["action"] = 3
    invalid_action["result"]["seed_action"] = 3
    with pytest.raises(ArtifactSchemaError, match="action.*range|range.*action"):
        document_to_result(invalid_action, connector)

    invalid_completeness, _, connector = make_document()
    invalid_completeness["result"]["completeness"]["radius_complete"] = False
    with pytest.raises(ArtifactSchemaError, match="minimal.*radius|radius.*minimal"):
        document_to_result(invalid_completeness, connector)

    invalid_existence, _, connector = make_document()
    invalid_existence["result"]["boundary_counterfactuals"] = []
    invalid_existence["result"]["minimal_counterfactuals"] = []
    invalid_existence["result"]["counterfactuals"] = {
        "minimal": [],
        "boundary": [],
    }
    with pytest.raises(ArtifactSchemaError, match="existence|found"):
        document_to_result(invalid_existence, connector)

    invalid_stats, _, connector = make_document()
    invalid_stats["result"]["stats"]["states_expanded"] = (
        invalid_stats["result"]["stats"]["states_evaluated"] + 1
    )
    with pytest.raises(ArtifactSchemaError, match="expanded.*evaluated|stats"):
        document_to_result(invalid_stats, connector)


@pytest.mark.parametrize("best_known_radius", [-7, 1])
def test_unknown_existence_requires_no_best_known_radius(
    best_known_radius: int,
) -> None:
    connector = ArtifactToyConnector()
    partial = compute_rr(
        "s",
        connector,
        ToyOracle(connector.space.actions, fingerprint="toy-policy-sha256"),
        SearchOptions(max_expanded=0),
    )
    document = result_to_document(partial, connector)
    document["result"]["best_known_radius"] = best_known_radius

    with pytest.raises(ArtifactSchemaError, match="unknown|non-negative"):
        document_to_result(document, connector)


def test_failed_write_never_publishes_target_or_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, connector = exact_result()
    target = tmp_path / "result.yaml"

    def fail_dump(*args: object, **kwargs: object) -> str:
        raise RuntimeError("simulated serialization interruption")

    monkeypatch.setattr(artifacts.yaml, "safe_dump", fail_dump)
    with pytest.raises((ArtifactError, RuntimeError), match="serialization interruption"):
        save_result(target, result, connector)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_safe_loader_rejects_python_specific_yaml_tags(tmp_path: Path) -> None:
    target = tmp_path / "unsafe.yaml"
    target.write_text(
        "schema: stache.rr-result\n"
        "schema_version: 1\n"
        "result: !!python/tuple [seed]\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactSchemaError, match="YAML|tag|construct"):
        load_result(target, ArtifactToyConnector())


@pytest.mark.parametrize("location", ["root", "nested"])
def test_artifact_yaml_loader_rejects_duplicate_mapping_keys(
    tmp_path: Path,
    location: str,
) -> None:
    document, _, _ = make_document()
    serialized = yaml.safe_dump(document, sort_keys=False)
    if location == "root":
        serialized = "schema: attacker.schema\n" + serialized
    else:
        serialized = serialized.replace(
            "  fingerprint:",
            "  fingerprint: attacker-policy\n  fingerprint:",
            1,
        )
    target = tmp_path / f"duplicate-{location}.yaml"
    target.write_text(serialized, encoding="utf-8")

    with pytest.raises(ArtifactSchemaError, match="duplicate"):
        load_result(target, ArtifactToyConnector())


def test_provenance_must_also_be_primitive() -> None:
    result, connector = exact_result()

    with pytest.raises(ArtifactError, match="provenance|primitive"):
        result_to_document(result, connector, provenance={"bad": ("tuple",)})


def test_input_document_is_not_mutated_during_load() -> None:
    document, _, connector = make_document()
    before = deepcopy(document)

    document_to_result(document, connector)

    assert document == before
