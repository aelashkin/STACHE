"""Behavioral contract tests for the domain-neutral RR/CF search."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import stache.explainability.core.search as search_module

from stache.explainability.core.connector import ExactActionInvariance
from stache.explainability.core.policy import TableActionOracle
from stache.explainability.core.models import (
    ContinuationMismatchError,
    CounterfactualExistence,
    CounterfactualSelection,
    InvalidSearchOptions,
    MetricCertificationError,
    MinimumBasis,
    SearchExtent,
    SearchInvariantError,
    SearchOptions,
    StopReason,
)
from stache.explainability.core.search import compute_rr

from ._toy import (
    ToyConnector,
    ToyOracle,
    brute_force,
    disconnected_formal_minimum_space,
    disconnected_unknown_space,
    exact_space,
    no_counterfactual_space,
    non_geodesic_space,
    query_budget_space,
    tied_minimum_space,
)


class VariableCostToyOracle(ToyOracle):
    """Test oracle whose successful uncached calls consume declared costs."""

    def __init__(
        self,
        actions: dict[str, int],
        costs: dict[str, int],
        *,
        declared_costs: dict[str, int] | None = None,
    ) -> None:
        super().__init__(actions)
        self._costs = costs
        self._declared_costs = costs if declared_costs is None else declared_costs

    def policy_query_cost(self, state: str) -> int:
        if state in self._cache:
            return 0
        return self._declared_costs[state]

    def action(self, state: str) -> int:
        if state in self._cache:
            return super().action(state)
        action = super().action(state)
        self._policy_queries += self._costs[state] - 1
        return action


def record_keys(records: object) -> tuple[str, ...]:
    return tuple(record.key for record in records)  # type: ignore[union-attr]


def record_signature(records: object) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            record.state,
            record.key,
            record.action,
            record.graph_depth,
            record.formal_distance,
        )
        for record in records  # type: ignore[union-attr]
    )


def semantic_signature(result: object) -> tuple[object, ...]:
    return (
        record_signature(result.region),  # type: ignore[union-attr]
        record_signature(result.boundary_counterfactuals),  # type: ignore[union-attr]
        record_signature(result.minimal_counterfactuals),  # type: ignore[union-attr]
        result.robustness_radius,  # type: ignore[union-attr]
        result.best_known_radius,  # type: ignore[union-attr]
        result.counterfactual_existence,  # type: ignore[union-attr]
        result.completeness.region_complete,  # type: ignore[union-attr]
        result.completeness.boundary_complete,  # type: ignore[union-attr]
        result.completeness.radius_complete,  # type: ignore[union-attr]
        result.completeness.minimal_counterfactuals_complete,  # type: ignore[union-attr]
    )


def exact_options(
    *,
    selection: CounterfactualSelection = CounterfactualSelection.BOTH,
    basis: MinimumBasis = MinimumBasis.GRAPH_BOUNDARY,
    max_expanded: int | None = None,
    max_policy_queries: int | None = None,
    max_graph_depth: int | None = None,
) -> SearchOptions:
    return SearchOptions(
        counterfactuals=selection,
        minimum_basis=basis,
        extent=SearchExtent.EXACT,
        max_expanded=max_expanded,
        max_policy_queries=max_policy_queries,
        max_graph_depth=max_graph_depth,
    )


def test_exact_search_matches_independent_fixed_point_oracle() -> None:
    space = exact_space()
    expected = brute_force(space, "s")

    result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        exact_options(),
    )

    assert frozenset(record_keys(result.region)) == expected.region
    assert frozenset(record_keys(result.boundary_counterfactuals)) == expected.boundary
    assert frozenset(record_keys(result.minimal_counterfactuals)) == expected.graph_minimal
    assert result.robustness_radius == expected.graph_radius
    assert result.counterfactual_existence is CounterfactualExistence.FOUND
    assert result.completeness.region_complete
    assert result.completeness.boundary_complete
    assert result.completeness.radius_complete
    assert result.completeness.minimal_counterfactuals_complete
    assert result.completeness.stop_reason is StopReason.COMPLETE
    assert result.continuation is None


@pytest.mark.parametrize(
    ("selection", "minimal_present", "boundary_present"),
    [
        (CounterfactualSelection.MINIMAL, True, False),
        (CounterfactualSelection.BOUNDARY, False, True),
        (CounterfactualSelection.BOTH, True, True),
    ],
)
def test_counterfactual_selection_is_an_explicit_output_projection(
    selection: CounterfactualSelection,
    minimal_present: bool,
    boundary_present: bool,
) -> None:
    space = exact_space()

    result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        exact_options(selection=selection),
    )

    assert (result.counterfactuals.minimal is not None) is minimal_present
    assert (result.counterfactuals.boundary is not None) is boundary_present
    if minimal_present:
        assert result.counterfactuals.minimal == result.minimal_counterfactuals
    if boundary_present:
        assert result.counterfactuals.boundary == result.boundary_counterfactuals


def test_exact_search_returns_every_tied_minimum_and_later_boundary() -> None:
    space = tied_minimum_space()
    expected = brute_force(space, "s")

    result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        exact_options(),
    )

    assert frozenset(record_keys(result.minimal_counterfactuals)) == {"x", "y"}
    assert frozenset(record_keys(result.minimal_counterfactuals)) == expected.graph_minimal
    assert frozenset(record_keys(result.boundary_counterfactuals)) == {"w", "x", "y"}
    assert result.robustness_radius == 2.0


def test_through_minimal_finishes_the_entire_layer_including_invariant_states() -> None:
    space = tied_minimum_space()
    options = SearchOptions(
        counterfactuals=CounterfactualSelection.MINIMAL,
        minimum_basis=MinimumBasis.GRAPH_BOUNDARY,
        extent=SearchExtent.THROUGH_MINIMAL_CF,
    )

    result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        options,
    )

    assert record_keys(result.region) == ("s", "a", "b", "c", "z")
    assert record_keys(result.minimal_counterfactuals) == ("x", "y")
    assert "w" not in record_keys(result.boundary_counterfactuals)
    assert result.completeness.radius_complete
    assert result.completeness.minimal_counterfactuals_complete
    assert not result.completeness.region_complete
    assert not result.completeness.boundary_complete
    assert result.completeness.remaining_frontier_size == 1
    assert result.completeness.stop_reason is StopReason.THROUGH_MINIMAL
    assert result.continuation is None


def test_through_minimal_is_complete_when_minimum_layer_exhausts_region() -> None:
    space = query_budget_space()
    options = SearchOptions(
        counterfactuals=CounterfactualSelection.MINIMAL,
        minimum_basis=MinimumBasis.GRAPH_BOUNDARY,
        extent=SearchExtent.THROUGH_MINIMAL_CF,
    )

    result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        options,
    )

    assert record_keys(result.region) == ("s",)
    assert record_keys(result.minimal_counterfactuals) == ("a", "b", "c")
    assert result.completeness.region_complete
    assert result.completeness.boundary_complete
    assert result.completeness.radius_complete
    assert result.completeness.minimal_counterfactuals_complete
    assert result.completeness.remaining_frontier_size == 0
    assert result.completeness.stop_reason is StopReason.COMPLETE
    assert result.continuation is None


def test_constant_policy_proves_counterfactual_absence_only_after_exact_completion() -> None:
    space = no_counterfactual_space()

    result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        exact_options(),
    )

    assert record_keys(result.region) == ("s", "a", "b")
    assert result.boundary_counterfactuals == ()
    assert result.minimal_counterfactuals == ()
    assert result.robustness_radius is None
    assert result.counterfactual_existence is CounterfactualExistence.PROVEN_ABSENT
    assert result.completeness.region_complete
    assert result.completeness.boundary_complete
    assert result.completeness.radius_complete
    assert result.completeness.minimal_counterfactuals_complete


def test_disconnected_same_action_island_does_not_enter_rr_or_prove_global_absence() -> None:
    space = disconnected_unknown_space()

    result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        exact_options(),
    )

    assert record_keys(result.region) == ("s", "a")
    assert result.boundary_counterfactuals == ()
    assert "i" not in record_keys(result.region)
    assert result.counterfactual_existence is CounterfactualExistence.UNKNOWN
    assert result.completeness.region_complete
    assert result.completeness.boundary_complete


def test_diamond_is_visited_once_and_output_order_ignores_neighbor_iteration_order() -> None:
    space = exact_space()
    forward_oracle = ToyOracle(space.actions)
    reverse_oracle = ToyOracle(space.actions)

    forward = compute_rr(
        "s", ToyConnector(space), forward_oracle, exact_options()
    )
    reverse = compute_rr(
        "s",
        ToyConnector(space, reverse_neighbors=True),
        reverse_oracle,
        exact_options(),
    )

    assert record_signature(forward.region) == record_signature(reverse.region)
    assert record_signature(forward.boundary_counterfactuals) == record_signature(
        reverse.boundary_counterfactuals
    )
    assert forward_oracle.calls.count("c") == 1
    assert reverse_oracle.calls.count("c") == 1
    assert forward_oracle.stats.policy_queries == len(space.states)
    assert reverse_oracle.stats.policy_queries == len(space.states)


def test_zero_expansion_still_validates_and_queries_seed_then_returns_frontier() -> None:
    space = exact_space()
    oracle = ToyOracle(space.actions)

    result = compute_rr(
        " S ",
        ToyConnector(space),
        oracle,
        exact_options(max_expanded=0),
    )

    assert result.seed.state == "s"
    assert record_keys(result.region) == ("s",)
    assert oracle.calls == ["s"]
    assert not result.completeness.region_complete
    assert not result.completeness.boundary_complete
    assert result.completeness.remaining_frontier_size == 1
    assert result.counterfactual_existence is CounterfactualExistence.UNKNOWN
    assert result.completeness.stop_reason is StopReason.MAX_EXPANDED
    assert result.continuation is not None


def test_invalid_seed_is_rejected_before_any_policy_query() -> None:
    space = exact_space()
    oracle = ToyOracle(space.actions)

    with pytest.raises(ValueError, match="unknown toy state"):
        compute_rr(
            "not-a-state",
            ToyConnector(space),
            oracle,
            exact_options(),
        )

    assert oracle.stats.policy_queries == 0


def test_query_budget_can_certify_radius_before_all_tied_minima() -> None:
    space = query_budget_space()

    result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        exact_options(max_policy_queries=2),
    )

    assert record_keys(result.minimal_counterfactuals) == ("a",)
    assert result.robustness_radius == 1.0
    assert result.best_known_radius == 1.0
    assert result.counterfactual_existence is CounterfactualExistence.FOUND
    assert result.completeness.radius_complete
    assert not result.completeness.minimal_counterfactuals_complete
    assert not result.completeness.region_complete
    assert not result.completeness.boundary_complete
    assert result.completeness.remaining_frontier_size == 2
    assert result.completeness.stop_reason is StopReason.MAX_POLICY_QUERIES
    assert result.continuation is not None


def test_query_budget_preflights_variable_cost_before_candidate_call() -> None:
    space = query_budget_space()
    oracle = VariableCostToyOracle(
        dict(space.actions),
        {"s": 1, "a": 2, "b": 1, "c": 1},
    )

    result = compute_rr(
        "s",
        ToyConnector(space),
        oracle,
        exact_options(max_policy_queries=2),
    )

    assert oracle.calls == ["s"]
    assert result.stats.policy_queries == 1
    assert result.stats.policy_queries <= 2
    assert result.completeness.stop_reason is StopReason.MAX_POLICY_QUERIES


def test_search_rejects_oracle_whose_actual_query_delta_differs_from_cost() -> None:
    space = query_budget_space()
    oracle = VariableCostToyOracle(
        dict(space.actions),
        {"s": 2, "a": 1, "b": 1, "c": 1},
        declared_costs={"s": 1, "a": 1, "b": 1, "c": 1},
    )

    with pytest.raises(SearchInvariantError, match="policy_query_cost"):
        compute_rr(
            "s",
            ToyConnector(space),
            oracle,
            exact_options(max_policy_queries=2),
        )


def test_zero_query_budget_rejects_uncached_seed_before_action_call() -> None:
    space = exact_space()
    oracle = ToyOracle(space.actions)

    with pytest.raises(InvalidSearchOptions, match="seed"):
        compute_rr(
            "s",
            ToyConnector(space),
            oracle,
            exact_options(max_policy_queries=0),
        )

    assert oracle.calls == []


def test_prewarmed_seed_costs_zero_and_preserves_zero_budget() -> None:
    space = exact_space()
    oracle = ToyOracle(space.actions)
    assert oracle.action("s") == space.actions["s"]

    result = compute_rr(
        "s",
        ToyConnector(space),
        oracle,
        exact_options(max_policy_queries=0),
    )

    assert oracle.calls == ["s"]
    assert result.stats.policy_queries == 0
    assert result.stats.cache_hits == 1
    assert result.completeness.stop_reason is StopReason.MAX_POLICY_QUERIES


def test_certified_formal_query_budget_preserves_observed_minimum() -> None:
    space = query_budget_space()

    result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        exact_options(
            basis=MinimumBasis.FORMAL_GLOBAL,
            max_policy_queries=2,
        ),
    )

    assert record_keys(result.minimal_counterfactuals) == ("a",)
    assert result.robustness_radius == 1.0
    assert result.best_known_radius == 1.0
    assert result.counterfactual_existence is CounterfactualExistence.FOUND
    assert result.completeness.radius_complete
    assert not result.completeness.minimal_counterfactuals_complete
    assert result.completeness.stop_reason is StopReason.MAX_POLICY_QUERIES


def test_certified_formal_through_minimal_returns_every_tied_minimum() -> None:
    space = query_budget_space()
    options = SearchOptions(
        counterfactuals=CounterfactualSelection.MINIMAL,
        minimum_basis=MinimumBasis.FORMAL_GLOBAL,
        extent=SearchExtent.THROUGH_MINIMAL_CF,
    )

    result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        options,
    )

    assert record_keys(result.minimal_counterfactuals) == ("a", "b", "c")
    assert result.robustness_radius == 1.0
    assert result.completeness.radius_complete
    assert result.completeness.minimal_counterfactuals_complete
    assert result.completeness.stop_reason is StopReason.COMPLETE


def test_graph_depth_budget_is_not_treated_as_a_formal_distance_result() -> None:
    space = exact_space()

    result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        exact_options(max_graph_depth=1),
    )

    assert record_keys(result.region) == ("s", "a", "b")
    assert result.boundary_counterfactuals == ()
    assert result.completeness.max_evaluated_graph_depth == 1
    assert not result.completeness.radius_complete
    assert result.counterfactual_existence is CounterfactualExistence.UNKNOWN
    assert result.completeness.remaining_frontier_size > 0
    assert result.completeness.stop_reason is StopReason.MAX_GRAPH_DEPTH
    assert result.continuation is not None


def test_continuation_resume_matches_uninterrupted_scientific_result() -> None:
    space = exact_space()
    connector = ToyConnector(space)

    first = compute_rr(
        "s",
        connector,
        ToyOracle(space.actions),
        exact_options(max_expanded=1),
    )
    assert first.continuation is not None

    second = compute_rr(
        "s",
        connector,
        ToyOracle(space.actions),
        exact_options(max_expanded=3),
        continuation=first.continuation,
    )
    assert second.continuation is not None

    resumed = compute_rr(
        "s",
        connector,
        ToyOracle(space.actions),
        exact_options(),
        continuation=second.continuation,
    )
    uninterrupted = compute_rr(
        "s",
        connector,
        ToyOracle(space.actions),
        exact_options(),
    )

    assert semantic_signature(resumed) == semantic_signature(uninterrupted)
    assert resumed.stats.policy_queries == uninterrupted.stats.policy_queries
    assert resumed.completeness.stop_reason is StopReason.COMPLETE
    assert resumed.continuation is None


def test_changed_table_content_cannot_resume_under_the_same_declared_label() -> None:
    space = exact_space()
    connector = ToyConnector(space)
    first = compute_rr(
        "s",
        connector,
        TableActionOracle(
            connector,
            space.actions,
            source_fingerprint="shared-label",
        ),
        exact_options(max_expanded=0),
    )
    assert first.continuation is not None
    changed_actions = {state: 1 for state in space.states}

    with pytest.raises(ContinuationMismatchError, match="fingerprint"):
        compute_rr(
            "s",
            connector,
            TableActionOracle(
                connector,
                changed_actions,
                source_fingerprint="shared-label",
            ),
            exact_options(),
            continuation=first.continuation,
        )


def test_builtin_oracle_continuation_supports_custom_hashable_state_keys() -> None:
    space = exact_space()

    class ObjectKey:
        def __init__(self, name: str) -> None:
            self.name = name

        def __hash__(self) -> int:
            return hash(self.name)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, ObjectKey) and self.name == other.name

    class ObjectKeyConnector(ToyConnector):
        def state_key(self, state: str) -> ObjectKey:
            return ObjectKey(state)

        def ordering_key(self, key: ObjectKey) -> tuple[str]:
            return (key.name,)

    connector = ObjectKeyConnector(space)
    partial = compute_rr(
        "s",
        connector,
        TableActionOracle(connector, space.actions),
        exact_options(max_expanded=0),
    )
    assert partial.continuation is not None

    resumed = compute_rr(
        "s",
        connector,
        TableActionOracle(connector, space.actions),
        exact_options(),
        continuation=partial.continuation,
    )

    assert resumed.counterfactual_existence is CounterfactualExistence.FOUND
    assert resumed.completeness.stop_reason is StopReason.COMPLETE

    corrupted = compute_rr(
        "s",
        connector,
        TableActionOracle(connector, space.actions),
        exact_options(max_expanded=0),
    )
    assert corrupted.continuation is not None
    corrupted.continuation.checkpoint.seed_key.name = "tampered"
    with pytest.raises(ContinuationMismatchError, match="payload|integrity"):
        compute_rr(
            "s",
            connector,
            TableActionOracle(connector, space.actions),
            exact_options(),
            continuation=corrupted.continuation,
        )


@pytest.mark.parametrize("field", ["checkpoint_version", "fingerprint"])
def test_continuation_rejects_tampered_version_or_fingerprint(field: str) -> None:
    space = exact_space()
    options = exact_options(max_expanded=1)
    partial = compute_rr(
        "s", ToyConnector(space), ToyOracle(space.actions), options
    )
    assert partial.continuation is not None
    tampered = replace(partial.continuation, **{field: "tampered"})

    with pytest.raises(ContinuationMismatchError, match=field):
        compute_rr(
            "s",
            ToyConnector(space),
            ToyOracle(space.actions),
            exact_options(),
            continuation=tampered,
        )


def test_continuation_rejects_tampered_checkpoint_payload() -> None:
    space = exact_space()
    partial = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        exact_options(max_expanded=1),
    )
    assert partial.continuation is not None
    checkpoint = deepcopy(partial.continuation.checkpoint)
    checkpoint.current_depth = 999
    tampered = replace(partial.continuation, checkpoint=checkpoint)

    with pytest.raises(ContinuationMismatchError, match="payload|integrity"):
        compute_rr(
            "s",
            ToyConnector(space),
            ToyOracle(space.actions),
            exact_options(),
            continuation=tampered,
        )


def test_continuation_digest_distinguishes_list_from_equal_tuple() -> None:
    space = exact_space()
    connector = ToyConnector(space)
    partial = compute_rr(
        "s",
        connector,
        ToyOracle(space.actions),
        exact_options(max_expanded=1),
    )
    assert partial.continuation is not None
    checkpoint = deepcopy(partial.continuation.checkpoint)
    checkpoint.current_layer = tuple(checkpoint.current_layer)
    assert (
        search_module._checkpoint_digest(checkpoint)
        != partial.continuation.payload_digest
    )
    tampered = replace(
        partial.continuation,
        checkpoint=checkpoint,
        payload_digest=search_module._checkpoint_digest(checkpoint),
    )

    with pytest.raises(ContinuationMismatchError, match="checkpoint|current_layer"):
        compute_rr(
            "s",
            connector,
            ToyOracle(space.actions),
            exact_options(),
            continuation=tampered,
        )


def test_continuation_revalidates_every_restored_state_key_binding() -> None:
    space = exact_space()
    connector = ToyConnector(space)
    partial = compute_rr(
        "s",
        connector,
        ToyOracle(space.actions),
        exact_options(max_expanded=1),
    )
    assert partial.continuation is not None
    checkpoint = deepcopy(partial.continuation.checkpoint)
    checkpoint.states["a"] = " A "
    tampered = replace(
        partial.continuation,
        checkpoint=checkpoint,
        payload_digest=search_module._checkpoint_digest(checkpoint),
    )

    with pytest.raises(ContinuationMismatchError, match="canonical|state.*key"):
        compute_rr(
            "s",
            connector,
            ToyOracle(space.actions),
            exact_options(),
            continuation=tampered,
        )


def test_non_geodesic_connector_separates_graph_and_formal_minimum_bases() -> None:
    space = non_geodesic_space()
    expected = brute_force(space, "s")

    graph_result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        exact_options(basis=MinimumBasis.GRAPH_BOUNDARY),
    )
    formal_result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        exact_options(basis=MinimumBasis.FORMAL_GLOBAL),
    )

    assert record_keys(graph_result.minimal_counterfactuals) == ("g",)
    assert graph_result.robustness_radius == expected.graph_radius == 3.0
    assert record_keys(formal_result.minimal_counterfactuals) == ("h",)
    assert formal_result.robustness_radius == expected.formal_radius == 2.0


def test_formal_layers_can_find_a_global_minimum_without_a_graph_depth() -> None:
    space = disconnected_formal_minimum_space()

    result = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        exact_options(basis=MinimumBasis.FORMAL_GLOBAL),
    )

    assert record_keys(result.boundary_counterfactuals) == ("g",)
    assert record_keys(result.minimal_counterfactuals) == ("q",)
    assert result.minimal_counterfactuals[0].graph_depth is None
    assert result.robustness_radius == 1.0
    assert result.completeness.radius_complete
    assert result.completeness.minimal_counterfactuals_complete
    assert result.stats.states_discovered == 4
    assert result.stats.states_evaluated == 4
    assert result.stats.policy_queries == 4


def test_formal_global_rejects_uncertified_connector_without_layers() -> None:
    space = non_geodesic_space(provide_formal_layers=False)

    with pytest.raises(MetricCertificationError, match="formal"):
        compute_rr(
            "s",
            ToyConnector(space),
            ToyOracle(space.actions),
            exact_options(basis=MinimumBasis.FORMAL_GLOBAL),
        )


def test_non_geodesic_formal_global_through_minimal_is_rejected() -> None:
    space = non_geodesic_space(provide_formal_layers=True)
    options = SearchOptions(
        counterfactuals=CounterfactualSelection.MINIMAL,
        minimum_basis=MinimumBasis.FORMAL_GLOBAL,
        extent=SearchExtent.THROUGH_MINIMAL_CF,
    )

    with pytest.raises(InvalidSearchOptions, match="through_minimal"):
        compute_rr("s", ToyConnector(space), ToyOracle(space.actions), options)


def test_phase_one_rejects_custom_invariance_semantics() -> None:
    space = exact_space()

    with pytest.raises(InvalidSearchOptions, match="exact action"):
        compute_rr(
            "s",
            ToyConnector(space),
            ToyOracle(space.actions),
            exact_options(),
            invariance=lambda _seed, _candidate: True,
        )

    explicit_exact = compute_rr(
        "s",
        ToyConnector(space),
        ToyOracle(space.actions),
        exact_options(),
        invariance=ExactActionInvariance(),
    )
    assert explicit_exact.counterfactual_existence is CounterfactualExistence.FOUND


def test_search_rejects_out_of_range_actions_from_custom_oracle() -> None:
    space = exact_space()
    actions = dict(space.actions)
    actions["a"] = max(space.actions.values()) + 1

    with pytest.raises(SearchInvariantError, match="declared range.*a"):
        compute_rr(
            "s",
            ToyConnector(space),
            ToyOracle(actions),
            exact_options(),
        )


def test_asymmetric_graph_exhaustion_does_not_prove_global_absence() -> None:
    space = no_counterfactual_space()

    class SeedCannotReachConnector(ToyConnector):
        def __init__(self) -> None:
            super().__init__(space)
            self.metric_certificate = replace(
                self.metric_certificate,
                symmetric=False,
                geodesic_for_formal_metric=False,
            )

        def atomic_neighbors(self, state: str) -> tuple[str, ...]:
            if state == "s":
                return ()
            if state == "a":
                return ("s",)
            return ("a",)

    actions = dict(space.actions)
    actions["a"] = 1

    result = compute_rr(
        "s",
        SeedCannotReachConnector(),
        ToyOracle(actions),
        exact_options(),
    )

    assert result.counterfactual_existence is CounterfactualExistence.UNKNOWN
    assert not result.completeness.radius_complete
    assert not result.completeness.minimal_counterfactuals_complete


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "counterfactuals": CounterfactualSelection.BOUNDARY,
            "minimum_basis": MinimumBasis.GRAPH_BOUNDARY,
            "extent": SearchExtent.THROUGH_MINIMAL_CF,
        },
        {
            "counterfactuals": CounterfactualSelection.BOTH,
            "minimum_basis": MinimumBasis.GRAPH_BOUNDARY,
            "extent": SearchExtent.THROUGH_MINIMAL_CF,
        },
        {"max_expanded": -1},
        {"max_expanded": True},
        {"max_policy_queries": -1},
        {"max_policy_queries": False},
        {"max_graph_depth": -1},
        {"max_graph_depth": True},
    ],
)
def test_invalid_option_combinations_fail_before_search(kwargs: dict[str, object]) -> None:
    with pytest.raises(InvalidSearchOptions):
        SearchOptions(**kwargs)  # type: ignore[arg-type]
