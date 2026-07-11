# Phase 1 task checklist

- [x] P0 — restore collection and installed-package/CLI baseline
  - Acceptance: all legacy test modules collect with installed `stache.*` imports;
    global item deletion is removed; stale signatures are current.
  - Verify: `uv run --python 3.11 --extra test pytest --collect-only -q`.
- [x] P1 — implement immutable generic connector/result/search/continuation contracts
  - Acceptance: exact and through-minimal semantics, orthogonal budgets/completeness,
    tri-state existence, metric-certificate gates, deterministic resume.
  - Verify: focused toy-graph tests against an independent oracle.
- [x] P2 — implement cached model/table/table-then-model action oracles
  - Acceptance: strict scalar normalization, seed parity, cache/query counts, explicit
    miss/fallback behavior, model observation/action-space validation.
  - Verify: focused oracle tests including invalid and batched shapes.
- [x] P3 — implement and migrate the Taxi connector and consumers
  - Acceptance: 500-state thesis universe, connector laws/geodesy, one-hot/artifact
    round trips, brute-force policy equivalence, compatibility warning/shim, `P == D`
    visual panels, no policy re-query by renderers.
  - Verify: exhaustive Taxi suite plus committed-DQN sample parity.
- [x] P4 — implement safe artifacts, DQN compatibility, and `stache compute-rr`
  - Acceptance: primitive-only schema/provenance, schema/fingerprint mismatch errors,
    safe legacy Taxi config migration, validated relative paths/config, installed CLI.
  - Verify: artifact round trips; CLI help/invalid/config/table/model smoke; wheel install.
- [x] P5 — document architecture, migration, rollback, and MiniGrid deferral
  - Acceptance: ADR and focused README are consistent with implemented contracts.
  - Verify: documentation review against audit Section 3 and final diff.
- [x] Final quality gate
  - Acceptance: full tests/build/imports pass or every unavailable/external failure is
    reported exactly; no unrelated/generated files; every material fresh-review finding
    is resolved or documented.
- [x] Publish without merging
  - Acceptance: intentional commits pushed; PR targets `main`; checks inspected; PR
    remains unmerged.
