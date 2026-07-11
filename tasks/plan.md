# Phase 1 specification: generic RR/CF core and Taxi connector

## Objective

Land the smallest mergeable architecture foundation that computes thesis-defined
robustness regions and counterfactuals once, behind domain connectors, and proves
the design exhaustively for Taxi's 500-state thesis universe. This phase restores a
trustworthy test/CLI baseline and safe artifacts without choosing new MiniGrid
scientific semantics.

Normative order: user constraints and workspace rules; thesis Definitions 5.2–5.17
and Algorithms 1–2; audit Section 3/finding IDs; current source and artifacts; runtime
evidence. Experimental thesis prose does not override the formal definitions.

## Supported stack and commands

- Python 3.11 (the pinned Torch 2.2.2 wheel does not install on the host's default
  Python 3.13/3.14 interpreters).
- Gymnasium 1.0.0, Stable-Baselines3 2.4.1, NumPy 1.26.4, PyYAML 6.0.2.
- Collection: `uv run --python 3.11 --extra test pytest --collect-only -q`
- Tests: `uv run --python 3.11 --extra test pytest -q`
- Build: `uv run --python 3.11 --with build --no-project python -m build`
- CLI: `stache compute-rr --config <path>` after wheel/editable installation.

## Architecture decisions

1. One pure, domain-neutral BFS owns RR/boundary/minimum/budget semantics. It imports
   no concrete domain, Gymnasium, SB3, rendering, YAML, or persistence code.
2. Connectors own canonical state identity, strict validation, declared universe,
   deterministic atomic neighbors/order, formal metric/certificate, observation and
   table-key codecs, action metadata, and primitive state/key artifact codecs.
3. Oracles own exact scalar-action normalization, one cache shared by seed/graph/
   formal scans, and explicit `model`, `table`, or `table_then_model` behavior.
4. `graph_boundary` minimizes BFS hops in the policy-changing RR boundary.
   `formal_global` minimizes formal distance across the declared universe. Boundary
   inference is allowed only with a geodesic certificate; otherwise ordered formal
   layers are required. Non-geodesic `formal_global + through_minimal_cf` is rejected
   in Phase 1 because a sound partial RR cutoff is not defined by the thesis.
5. Results separately report region, boundary, radius, and all-tied-minima
   completeness; global existence is `found`, `proven_absent`, or `unknown`.
   Graph depth is optional for independently discovered formal-global states.
6. Budget stops are resumable through a versioned checkpoint containing ordered
   graph/formal phase state, accumulated records, visited canonical states, action
   cache, statistics, and a scientific fingerprint. Resource ceilings may increase
   on resume; scientific options may not change.
7. Taxi uses every `(row, column, passenger, destination)` tuple in
   `5 × 5 × 5 × 4 = 500`, including `P == D`. Atomic edges ignore road walls and use
   row/column ±1 plus any other passenger/destination category, matching the thesis.
8. New artifacts contain only recursively validated JSON/YAML primitives and have
   independent result, continuation, connector, metric, observation-codec, and
   state-codec versions. Existing Python-tagged Taxi configs are accepted only by a
   narrow warning-emitting compatibility reader and are normalized to primitives.

## Work packets

| Packet | Findings | Expected surface | Acceptance checkpoint |
|---|---|---|---|
| P0: baseline | T1, T2, Q1, G2 | tests, `pyproject.toml` | installed imports; all legacy modules collect; no global filtering; unsafe broken entry points are not exposed |
| P1: core | A1, C3, C7, C9, P1 | `explainability/core/{connector,models,search}.py` | toy exact/partial/budget/non-geodesic/continuation oracle tests; deterministic visited-on-enqueue BFS |
| P2: policy | A2, C4, C8, R2, R6 | `explainability/core/policy.py` | scalar-shape, cache, table miss/fallback, seed parity, model-space mismatch tests |
| P3: Taxi | C6, C8, A1/A2, T2/T4 | `connectors/taxi.py`, legacy shim, Taxi consumers | all 500 states, 100 `P == D`, exhaustive laws/geodesy, independent arbitrary-table oracle, committed-model sample parity |
| P4: artifacts/CLI | R1, R4, G2, G6, D1 | `artifacts.py`, `cli/`, experiment I/O | primitive round trips, mismatch rejection, DQN loading, installed `stache compute-rr`, invalid-config/help smoke |
| P5: decision record | D5, G5, G8 | ADR and focused README updates | semantics, versions, migration, rollback, and MiniGrid deferral are explicit |

## Testing strategy

- Tests are behavioral and independent of legacy RR YAML. Toy graphs use an explicit
  state list, edge set, formal-distance map, and fixed-point/brute-force oracle.
- Taxi expected neighbors and policy results are derived independently from the
  formal metric over all 500 states, not from `TaxiConnector.unit_neighbors`.
- Cheap committed-model tests load DQN only; no training or tuning is permitted.
- Every behavior change follows red → green → refactor, then an atomic commit after a
  green checkpoint. Final gates include collect-only, focused tests, full pytest,
  wheel install/import, CLI help/invalid arguments, and a five-axis diff review.

## Boundaries

- Always: strict input/state/action/version validation; deterministic output; explicit
  completeness; staged-diff and secret scans before each commit.
- Ask first: any MiniGrid state-universe/codec/neighbor/model-input change, CI change,
  dependency change, training/tuning, artifact regeneration, or history rewrite.
- Never: infer formal-global minima from an uncertified boundary; hide partial output;
  unsafe YAML loading; merge/deploy/train/tune; read reviewer-only materials.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Partial results look exact | orthogonal completeness flags, stop reason, continuation, artifact schema validation |
| Resuming changes science | fingerprint connector/policy/predicate/seed/semantic options; permit only monotonic ceiling increases |
| Table/model disagree at seed | one shared oracle for every query; document corrected legacy behavior |
| Non-geodesic overclaim | certificate gate or independent formal layers; reject unsound option combinations |
| Legacy artifacts contain Python tags | narrowly scoped tuple-tag compatibility loader; all new writes use primitive-safe dumping |
| PR expands into MiniGrid redesign | no MiniGrid semantic/source/artifact changes; ADR records the required follow-up decision sequence |

## Deliberately deferred

MiniGrid C1, C2, C4, C5, and the MiniGrid portion of C7; broad training/tuning and
evaluation cleanup; checkpoint regeneration; large artifact migration; a 404-state
Taxi variant; batching/caching performance work beyond the required correctness cache;
and unrelated entry-point/CI modernization. MiniGrid follow-up must first select a
state universe, historical observation codec, collision/cardinality rules, formal
minimum strategy, shadow-model checks, and artifact revalidation plan.

