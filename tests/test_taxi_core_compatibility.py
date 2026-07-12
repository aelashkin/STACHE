"""Taxi migration contracts for the generic RR core.

These tests keep the modern typed result separate from the deprecated mapping
shape used by the original Taxi scripts.  Policy expectations come from an
independent 500-entry table rather than legacy RR artifacts.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from stable_baselines3 import DQN
import yaml

from stache.explainability.artifacts import ArtifactError, load_result
from stache.explainability.connectors.taxi import TaxiConnector
from stache.explainability.core.models import SearchResult
from stache.explainability.core.policy import ModelManifest, normalize_discrete_action
from stache.explainability.taxi.robust_taxi import (
    compute_rr_taxi,
    compute_taxi_rr,
    get_neighbors_taxi,
    translate_tuple_to_onehot,
)
from stache.explainability.taxi.taxi_robustness_region_visualization import (
    _minimal_counterfactuals_for_plot,
    _taxi_panel_pairs,
)
from stache.explainability.taxi import taxi_robustness_region_visualization
from stache.explainability.taxi.taxi_policy_map import (
    ACTION_NAMES,
    _policy_map_panel_pairs,
    build_action_grid,
    collect_state_actions,
    plot_dest_maps,
    save_mapping_yaml,
)
from stache.explainability.taxi import taxi_policy_map


TaxiState = tuple[int, int, int, int]


class TableModel:
    """Small SB3-shaped policy double backed by a complete Taxi table."""

    observation_space = SimpleNamespace(shape=(500,), dtype=np.dtype("float32"))
    action_space = SimpleNamespace(n=6)

    def __init__(self, table: dict[int, int]) -> None:
        self.table = table
        self.calls: Counter[int] = Counter()

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        assert deterministic is True
        assert observation.shape == (500,)
        index = int(np.argmax(observation))
        self.calls[index] += 1
        return np.asarray([self.table[index]], dtype=np.int64), None


def _policy_table() -> dict[int, int]:
    return {
        index: (index * 37 + index // 13 + (index % 5) * 2) % 6
        for index in range(500)
    }


def _model_manifest(fingerprint: str) -> ModelManifest:
    connector = TaxiConnector()
    return ModelManifest(
        model_fingerprint=fingerprint,
        observation_identity=connector.observation_spec.identity,
        action_spec=connector.action_spec,
    )


def _scientific_signature(result: SearchResult[object, object]) -> tuple[object, ...]:
    return (
        result.seed.state,
        result.seed_action,
        tuple(
            (record.state, record.action, record.graph_depth, record.formal_distance)
            for record in result.region
        ),
        tuple(
            (record.state, record.action, record.graph_depth, record.formal_distance)
            for record in result.boundary_counterfactuals
        ),
        tuple(
            (record.state, record.action, record.graph_depth, record.formal_distance)
            for record in result.minimal_counterfactuals
        ),
        result.robustness_radius,
        result.counterfactual_existence,
        result.completeness,
    )


@pytest.fixture(scope="module")
def committed_policy() -> tuple[DQN, dict[int, int], str]:
    """Load one checked-in DQN and independently materialize its 500 actions."""

    path = Path("data/experiments/models/Taxi-v3_DQN_model_0/model.zip")
    assert path.is_file(), f"missing committed Taxi model: {path}"
    model = DQN.load(path, env=None)
    connector = TaxiConnector()
    table: dict[int, int] = {}
    for state in connector.declared_states():
        raw_action, _ = model.predict(
            connector.encode_observation(state),
            deterministic=True,
        )
        table[connector.encode_index(state)] = normalize_discrete_action(
            raw_action,
            connector.action_spec.count,
        )
    fingerprint = f"sha256:{sha256(path.read_bytes()).hexdigest()}"
    return model, table, fingerprint


def test_modern_compute_taxi_rr_returns_typed_generic_result() -> None:
    seed = (2, 2, 4, 1)
    result = compute_taxi_rr(seed, policy_table=_policy_table())

    assert isinstance(result, SearchResult)
    assert result.seed.state == seed
    assert result.metadata.connector_identity.domain == "taxi"
    assert result.metadata.connector_identity.state_universe == "taxi-factored-500"


def test_model_and_complete_table_sources_have_identical_scientific_results() -> None:
    table = _policy_table()
    seed = (2, 2, 4, 1)

    table_result = compute_taxi_rr(seed, policy_table=table)
    model_result = compute_taxi_rr(
        seed,
        model=TableModel(table),
        model_fingerprint="test-model:same-policy-v1",
        model_manifest=_model_manifest("test-model:same-policy-v1"),
    )

    assert _scientific_signature(model_result) == _scientific_signature(table_result)


@pytest.mark.parametrize("seed", [(0, 0, 0, 0), (4, 4, 4, 3)])
def test_committed_dqn_and_materialized_table_have_seed_and_result_parity(
    committed_policy: tuple[DQN, dict[int, int], str],
    seed: TaxiState,
) -> None:
    model, table, fingerprint = committed_policy

    table_result = compute_taxi_rr(seed, policy_table=table)
    model_result = compute_taxi_rr(
        seed,
        model=model,
        model_fingerprint=fingerprint,
        model_manifest=_model_manifest(fingerprint),
    )

    assert model_result.seed_action == table_result.seed_action
    assert _scientific_signature(model_result) == _scientific_signature(table_result)


def test_table_then_model_source_uses_the_table_for_the_seed() -> None:
    connector = TaxiConnector()
    seed = (0, 0, 0, 0)
    seed_index = connector.encode_index(seed)
    model_table = {index: 1 for index in range(500)}
    model = TableModel(model_table)

    result = compute_taxi_rr(
        seed,
        model=model,
        policy_table={seed_index: 0},
        model_fingerprint="test-model:fallback-v1",
        model_manifest=_model_manifest("test-model:fallback-v1"),
    )

    assert result.seed_action == 0
    assert model.calls[seed_index] == 0
    assert result.stats.table_hits == 1
    assert result.stats.model_queries > 0


def test_legacy_compute_rr_taxi_warns_and_retains_mapping_contract() -> None:
    connector = TaxiConnector()
    table = _policy_table()
    seed = (1, 3, 2, 0)
    model = TableModel(table)
    manifest = _model_manifest("legacy-table-model-v1")
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            observation_space=SimpleNamespace(n=500),
        )
    )

    with pytest.warns(DeprecationWarning, match="compute_taxi_rr"):
        legacy = compute_rr_taxi(
            seed,
            model,
            env,
            precomputed_sa=table,
            model_manifest=manifest,
        )

    assert set(legacy) == {
        "rr_tuple_set",
        "rr_depths",
        "counterfactuals",
        "initial_action",
        "stats",
    }
    assert isinstance(legacy["rr_tuple_set"], set)
    assert all(isinstance(state, tuple) and len(state) == 4 for state in legacy["rr_tuple_set"])
    assert isinstance(legacy["rr_depths"], dict)
    assert all(type(depth) is int for depth in legacy["rr_depths"].values())
    assert all(
        isinstance(item, tuple)
        and len(item) == 3
        and isinstance(item[0], tuple)
        and type(item[1]) is int
        and type(item[2]) is int
        for item in legacy["counterfactuals"]
    )
    assert type(legacy["initial_action"]) is int
    assert set(legacy["stats"]) == {
        "region_size",
        "visited",
        "opened",
        "elapsed",
    }
    assert legacy["initial_action"] == table[connector.encode_index(seed)]
    # The explicit table-then-model source must use the table even at the seed.
    assert model.calls[connector.encode_index(seed)] == 0


def test_legacy_compute_rr_taxi_requires_explicit_model_manifest() -> None:
    model = TableModel(_policy_table())
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            observation_space=SimpleNamespace(n=500),
        )
    )

    with pytest.warns(DeprecationWarning), pytest.raises(
        ValueError,
        match="model_manifest",
    ):
        compute_rr_taxi((0, 0, 0, 0), model, env)


def test_model_only_search_queries_the_seed_once() -> None:
    connector = TaxiConnector()
    table = {index: 0 for index in range(500)}
    seed = (4, 4, 4, 3)
    model = TableModel(table)

    result = compute_taxi_rr(
        seed,
        model=model,
        model_fingerprint="test-model:constant-v1",
        model_manifest=_model_manifest("test-model:constant-v1"),
    )

    assert len(result.region) == 500
    assert model.calls[connector.encode_index(seed)] == 1
    assert sum(model.calls.values()) == 500


def test_legacy_conversion_and_neighbor_helpers_delegate_to_connector() -> None:
    connector = TaxiConnector()
    state = (2, 2, 0, 1)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(observation_space=SimpleNamespace(n=500))
    )

    with pytest.warns(DeprecationWarning, match="TaxiConnector"):
        observation = translate_tuple_to_onehot(env, state)
    with pytest.warns(DeprecationWarning, match="TaxiConnector"):
        neighbors = get_neighbors_taxi(state)

    assert observation.shape == (1, 500)
    assert np.array_equal(observation[0], connector.encode_observation(state))
    assert neighbors == list(connector.atomic_neighbors(state))


def test_visualization_uses_result_actions_without_policy_requery() -> None:
    table = _policy_table()
    model = TableModel(table)
    result = compute_taxi_rr(
        (2, 2, 4, 1),
        model=model,
        model_fingerprint="test-model:visualization-v1",
        model_manifest=_model_manifest("test-model:visualization-v1"),
    )
    before = model.calls.copy()

    plotted = _minimal_counterfactuals_for_plot(result)

    assert model.calls == before
    assert plotted == tuple(
        (record.state, record.action)
        for record in result.minimal_counterfactuals
    )


def test_visualization_panels_cover_all_twenty_passenger_destination_pairs() -> None:
    panels = _taxi_panel_pairs()

    assert len(panels) == 20
    assert len(set(panels)) == 20
    assert set(panels) == {
        (passenger, destination)
        for destination in range(4)
        for passenger in range(5)
    }
    assert all((destination, destination) in panels for destination in range(4))


def test_visualization_cli_writes_canonical_artifact_and_requires_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = TaxiConnector()
    seed = (0, 0, 0, 2)
    result = compute_taxi_rr(seed, policy_table=_policy_table())
    model_dir = tmp_path / "model-under-test"
    model_dir.mkdir()
    (model_dir / "model.zip").write_bytes(b"test model snapshot")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        taxi_robustness_region_visualization,
        "load_trusted_taxi_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            model=object(),
            model_fingerprint="test-model:visualization-v1",
            manifest=_model_manifest("test-model:visualization-v1"),
        ),
    )
    monkeypatch.setattr(
        taxi_robustness_region_visualization,
        "compute_taxi_rr",
        lambda *_args, **_kwargs: result,
    )
    expected_provenance = {"dependencies": {"python": "test-version"}}
    monkeypatch.setattr(
        taxi_robustness_region_visualization,
        "collect_provenance",
        lambda: expected_provenance,
        raising=False,
    )

    arguments = [
        "--model-path",
        str(model_dir),
        "--state",
        "0,0,0,2",
        "--acknowledge-trusted-model",
        "--hide-walls",
    ]
    taxi_robustness_region_visualization.main(arguments)

    artifact_path = (
        tmp_path
        / "data"
        / "experiments"
        / "rr"
        / "taxi_robustness_region"
        / model_dir.name
        / "0_0_0_2"
        / "robustness_region.yaml"
    )
    document = yaml.safe_load(artifact_path.read_text(encoding="utf-8"))
    assert "rr_tuples" not in document
    assert document["provenance"] == expected_provenance
    assert load_result(
        artifact_path,
        connector,
        expected_policy_fingerprint=result.metadata.policy_fingerprint,
    ) == result

    with pytest.raises(ArtifactError, match="overwrite"):
        taxi_robustness_region_visualization.main(arguments)

    taxi_robustness_region_visualization.main([*arguments, "--overwrite"])


def test_visualizer_clis_require_trust_before_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accessed = False

    def unexpected_access(*_args: object, **_kwargs: object) -> object:
        nonlocal accessed
        accessed = True
        raise AssertionError("model access must follow trust acknowledgement")

    monkeypatch.setattr(
        taxi_robustness_region_visualization,
        "load_trusted_taxi_model",
        unexpected_access,
    )
    monkeypatch.setattr(
        taxi_policy_map,
        "load_trusted_taxi_model",
        unexpected_access,
    )

    with pytest.raises(SystemExit):
        taxi_robustness_region_visualization.main(
            ["--model-path", str(tmp_path)]
        )
    with pytest.raises(SystemExit, match="acknowledge-trusted-model"):
        taxi_policy_map.main(["--model-path", str(tmp_path)])

    assert accessed is False


def test_policy_map_rejects_unsafe_timestamp_before_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accessed = False

    def unexpected_access(*_args: object, **_kwargs: object) -> object:
        nonlocal accessed
        accessed = True
        raise AssertionError("invalid output path must fail before model access")

    monkeypatch.setattr(
        taxi_policy_map,
        "load_trusted_taxi_model",
        unexpected_access,
    )

    with pytest.raises(ValueError, match="timestamp"):
        taxi_policy_map.run_visualisation(
            tmp_path,
            timestamp="../outside",
            acknowledge_trusted_model=True,
        )

    assert accessed is False


def test_policy_map_collects_all_500_connector_states_once() -> None:
    connector = TaxiConnector()
    table = _policy_table()
    model = TableModel(table)

    mapping = collect_state_actions(
        model,
        connector=connector,
        model_fingerprint="test-model:policy-map-v1",
        model_manifest=_model_manifest("test-model:policy-map-v1"),
    )

    assert mapping == table
    assert set(mapping) == set(range(500))
    assert sum(model.calls.values()) == 500
    assert all(call_count == 1 for call_count in model.calls.values())
    assert connector.encode_index((0, 0, 0, 0)) in mapping
    assert connector.encode_index((4, 4, 3, 3)) in mapping


def test_policy_map_grid_uses_connector_index_and_supports_p_equal_d() -> None:
    connector = TaxiConnector()
    mapping = {index: index % 6 for index in range(500)}

    grid = build_action_grid(
        mapping,
        passenger_loc=2,
        dest_idx=2,
        connector=connector,
    )

    assert grid.shape == (5, 5)
    assert all(
        grid[row, column]
        == mapping[connector.encode_index((row, column, 2, 2))]
        for row in range(5)
        for column in range(5)
    )


def test_policy_map_metadata_and_panels_come_from_thesis_connector_view() -> None:
    connector = TaxiConnector()
    expected_names = {
        int(item["action"]): str(item["label"]).title()
        for item in connector.action_metadata
    }
    panels = _policy_map_panel_pairs()

    assert ACTION_NAMES == expected_names
    assert panels == tuple(
        (passenger, destination)
        for destination in (3, 1, 0, 2)
        for passenger in range(5)
    )
    assert len(panels) == 20
    assert all((destination, destination) in panels for destination in range(4))


def test_destination_plot_renders_all_five_passenger_factors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = TaxiConnector()
    mapping = _policy_table()
    calls: list[tuple[int, int]] = []
    real_build_action_grid = build_action_grid

    def recording_grid(
        policy_mapping: dict[int, int],
        passenger_loc: int,
        dest_idx: int,
        *,
        connector: TaxiConnector | None = None,
    ) -> np.ndarray:
        calls.append((passenger_loc, dest_idx))
        return real_build_action_grid(
            policy_mapping,
            passenger_loc,
            dest_idx,
            connector=connector,
        )

    monkeypatch.setattr(taxi_policy_map, "build_action_grid", recording_grid)
    output = tmp_path / "destination-2.png"

    plot_dest_maps(
        mapping,
        2,
        output,
        connector=connector,
        show_walls=False,
    )

    assert calls == [(passenger, 2) for passenger in range(5)]
    assert (2, 2) in calls
    assert output.is_file()


def test_committed_dqn_policy_map_matches_independent_materialization(
    committed_policy: tuple[DQN, dict[int, int], str],
) -> None:
    model, expected_table, fingerprint = committed_policy

    mapping = collect_state_actions(
        model,
        connector=TaxiConnector(),
        model_fingerprint=fingerprint,
        model_manifest=_model_manifest(fingerprint),
    )

    assert mapping == expected_table


def test_policy_map_legacy_env_first_calls_warn_and_delegate(
    tmp_path: Path,
) -> None:
    connector = TaxiConnector()
    table = _policy_table()
    model = TableModel(table)
    model.stache_model_manifest = _model_manifest("legacy-policy-map-v1")
    wrapped_env = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(500,))
    )
    base_env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            observation_space=SimpleNamespace(n=500)
        )
    )

    with pytest.warns(DeprecationWarning, match="env/base_env"):
        collected = collect_state_actions(model, wrapped_env, base_env)
    with pytest.warns(DeprecationWarning, match="taxi_env"):
        legacy_grid = build_action_grid(base_env, table, 2, 2)
    output = tmp_path / "legacy-destination.png"
    with pytest.warns(DeprecationWarning, match="taxi_env"):
        plot_dest_maps(base_env, table, 2, output, False)

    assert collected == table
    assert np.array_equal(
        legacy_grid,
        build_action_grid(
            table,
            2,
            2,
            connector=connector,
        ),
    )
    assert output.is_file()


def test_policy_map_yaml_is_safe_primitive_complete_mapping(
    tmp_path: Path,
) -> None:
    table = _policy_table()
    output = tmp_path / "state-action.yaml"

    save_mapping_yaml(table, output)

    serialized = output.read_text(encoding="utf-8")
    assert "!!python" not in serialized
    assert yaml.safe_load(serialized) == table
