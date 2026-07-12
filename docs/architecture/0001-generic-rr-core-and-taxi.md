# ADR 0001: Generic RR core and Taxi connector boundary

- Status: Accepted for Phase 1
- Date: 2026-07-11
- Revised: 2026-07-12

This ADR records the generic-core and Taxi decisions. It does not claim closure
of every audit finding that motivated the work. MiniGrid universe/codec work,
reproduction automation, CI, and broader repository-quality findings remain
partial or deferred as stated below and in the public migration guide.

## Context

STACHE previously had separate robustness-region traversals for MiniGrid and
Taxi. Those implementations mixed state identity, policy encoding, traversal,
rendering, and persistence, and their partial results did not carry enough
information to distinguish observations from certified scientific results.

The normative definitions are the thesis' formal state space and distance,
Robustness Region, Minimal Counterfactual, Algorithms 1 and 2, the composite
explanation framework, and the reconciled OO-MDP projection. A versioned
projection `φ` maps a raw state to canonical object/factor state. Numeric factors
use Manhattan distance and categorical factors use Hamming distance. Distinct
vertices are adjacent when `0 < d(φ(s), φ(s')) <= ε`; this system fixes
`ε = δ = 1`. In particular:

- the RR is the connected same-action component containing the seed;
- an atomic graph edge represents exactly one declared formal unit perturbation;
- the formal robustness radius is a global minimum over the declared state
  universe, including every tied minimum;
- Algorithm 2's graph-layer stopping rule establishes a formal-global claim
  only when the connector truthfully certifies the missing geodesic premise.

The architecture audit leaves MiniGrid's scientific state universe unresolved.
Changing its object order, observation encoding, or neighbor relation would
change model inputs and potentially invalidate published artifacts.

## Decision

### One domain-neutral search

`stache.explainability.core` owns the RR/CF search exactly once. It imports no
Taxi, MiniGrid, Gymnasium, Stable-Baselines3, rendering, YAML, or artifact code.
Connectors own canonical state identity, validation, stable keys and ordering,
the declared universe, atomic neighbors, formal distance, and the truth of a
versioned metric certificate.

The action-oracle boundary normalizes a policy output to one Python `int` in a
declared discrete action space and Phase 1 uses exact action equality as its only
invariance predicate. Seed and candidate queries use the same cache. The
supported sources are a strict table, a model, and explicit table-then-model
fallback. Table fallback therefore applies at the seed too. Automatically
derived table fingerprints bind the entries, action count, action-normalization
version, and error-versus-model-fallback policy. Caller labels are retained as
provenance but combined with the content digest; they never replace scientific
policy identity or permit a changed table to resume an old continuation.

Every oracle supplies a pure, exact `policy_query_cost(state)` preflight. A
cached action costs zero; otherwise the declared cost must equal the observed
policy-query counter delta. The core checks the ceiling before invoking the
source and raises an invariant error if the declaration and actual delta differ.
An uncached seed needs a positive query budget because RR membership cannot be
defined without `π(s0)`; a prewarmed seed can be evaluated with a zero ceiling.

### Graph and formal minima are distinct

The result records graph depth and formal distance independently.

| Minimum basis | Meaning | Permitted evidence |
| --- | --- | --- |
| `graph_boundary` | Smallest graph-hop depth among policy-changing RR boundary states | Complete BFS layers in the connector's atomic graph |
| `formal_global` | Smallest formal distance among every policy-changing state in the declared universe | A sufficient geodesic certificate, or connector-provided complete increasing formal-distance layers |

The core never infers a formal-global minimum from a graph boundary when those
conditions are absent. A formal-global minimum may therefore have no graph
depth when it lies outside the seed's graph component.

### Extent, budgets, and completeness

`exact` searches the complete connected RR and boundary. `through_minimal_cf`
finishes the first policy-changing graph layer, including invariant peers, and
is only valid for a minimal-counterfactual projection. Requested extent is
separate from the optional total ceilings `max_expanded`,
`max_policy_queries`, and `max_graph_depth`.

`states_expanded` counts states whose connector neighbors were enumerated.
`policy_queries` counts uncached source evaluations, not cache hits or table/model
subtype counters. Ceilings apply to cumulative totals across continuation.

Every result reports completeness independently for:

- the observed robustness region;
- the policy-changing graph boundary;
- the certified radius;
- all tied minima.

Counterfactual existence is `found`, `proven_absent`, or `unknown`. Mathematical
infinity is represented by `proven_absent` with a null serialized radius, not a
non-standard JSON infinity. A first counterfactual in a partly evaluated layer
can certify the radius while leaving tied-minimum completeness false.

Budget stops produce a continuation-v2 in-memory checkpoint with a semantic
fingerprint, cryptographic payload-integrity digest, ordered frontier, phase and
layer cursors, resolved actions, action-cache checkpoint, accumulated result,
and cumulative statistics. Its digest type-tags lists, tuples, sets, and frozen
sets, and resume fully revalidates canonical states/keys, ordering ownership,
frontier structure, counters, formal layers, and cache entries before reuse.
Scientific options cannot change on resume;
resource ceilings may be increased because they are deliberately excluded from
the semantic fingerprint.

Artifact v2 deliberately serializes only non-resumable checkpoint metadata. The
opaque in-memory continuation payload is not accepted from YAML.

### Taxi is the first connector

Taxi projects every raw `0..499` index to the canonical tuple
`(row, column, passenger, destination)` and uses the thesis' complete
`5 × 5 × 5 × 4 = 500` factored universe. The 100
states where passenger location equals destination (`P == D`) are mandatory.
Its formal metric is:

```text
|row1-row2| + |column1-column2| + [passenger differs] + [destination differs]
```

Atomic perturbations are row/column ±1 within bounds and categorical passenger
or destination changes to any other value. Taxi road walls are MDP transition
constraints and do not restrict this explanation graph. Exhaustive connector
tests establish symmetry, complete unit-neighbor coverage, connectivity, and
graph/formal geodesy over all 250,000 ordered state pairs.

The connector owns the 0..499 policy-table key, flat float32 500-way one-hot
model observation, action metadata, and primitive state/key codec. The one-hot
vector is a separately versioned model-observation identity, not a scientific
factor or distance input. Rendering is an optional consumer outside the core.

### Artifacts and compatibility

New RR result artifacts use artifact schema v2, core schema v2, and an
independently versioned connector codec. Documents contain only null, booleans,
finite numbers, strings, lists, and string-keyed mappings. They include
connector/universe/metric/codec versions, object projection, factorization,
topology, adjacency threshold, the metric certificate, model
observation/action identities, policy fingerprint/source, options, derived
completeness evidence, stop reason, statistics, and supplied Git/dependency
provenance.
Loading verifies schema, connector identity, expected policy fingerprint, and
lossless state/key round trips. It also recomputes connector-owned formal
distances, reconciles repeated state records, and, for complete graph results,
re-walks RR closure and BFS depths before accepting radius/minimum claims.
Current RR artifacts and `stache compute-rr` configuration/policy-table YAML
reject duplicate mapping keys recursively.

Artifact loading is structural and scientific-evidence validation, not policy
re-execution or authentication. In particular, the result schema does not carry
a full action ledger for every state scanned by a non-geodesic `formal_global`
search. Its all-tied-minima completeness flag is therefore a producer assertion:
the loader checks it against every serialized counterfactual record but cannot
detect a disconnected formal tie omitted from the document entirely. Consumers
of an untrusted artifact must verify the expected policy fingerprint through a
trusted channel and recompute the policy/search when that global assertion is
material.

Stable-Baselines3 archives are a trust boundary: loading can deserialize
cloudpickled Python objects. Every model-loading console path requires explicit
`--acknowledge-trusted-model`. The shared loader validates the sidecar against
the Taxi connector, reads an archive once, and passes the same immutable byte
snapshot to SHA-256 and `DQN.load`, preventing provenance from disagreeing with
loaded bytes. The fingerprint is an identity/integrity record, not a signature,
authenticity proof, or sandbox; users must load only trusted model archives.

Taxi training uses the connector's observation encoder for all 500 raw indices.
Saving a Taxi experiment atomically writes `model.manifest.yaml` beside
`model.zip`; the manifest binds the saved archive fingerprint, observation
identity, and six-action contract. Training occurs only from an explicit entry
point and never at module import.

`compute_rr_taxi` remains as a warning-emitting compatibility shim. Its legacy
dictionary shape remains available, while scientific computation is delegated
to the generic core. Model-backed calls must pass a verified `ModelManifest`
(or attach one as `stache_model_manifest`); guessing semantics from a live model
is no longer accepted. The corrected table-at-seed behavior is intentional and
documented because the old mixed source could report a seed action from the
model while using table actions for candidates. Python-tagged historical YAML
is not silently treated as the current safe result schema.

## Deliberately deferred MiniGrid decision

This ADR does not register a MiniGrid connector and does not choose its broader
object universe, observation/artifact codec, state injection, metric
certificate, models, or published-artifact migration. One narrow historical
neighbor rule is corrected and tested against MiniGrid 3.0.0: a direction is
adjacent only to the headings reached by one `left` or `right` action (±90°).
The opposite heading requires two turns and is not adjacent. Follow-up work for
C1, C2, C4, C5, and the MiniGrid portion of C7 must first:

1. choose and document the scientific state universe;
2. version both historical and proposed observation/artifact codecs;
3. run shadow action checks on committed models before changing inputs;
4. independently validate the neighbor graph and metric certificate;
5. revalidate, rather than regenerate by default, affected research artifacts;
6. migrate consumers only after compatibility and scientific review.

## Consequences and rollback

Partial results can no longer be mistaken for exact/global results, and Taxi
model/table policies share one reproducible path. The cost is an intentionally
stricter public contract:

- artifact-v1 results must be recomputed as artifact v2;
- continuation-v1 checkpoints cannot resume and must restart;
- custom action oracles must implement exact `policy_query_cost`;
- model-backed APIs and the Taxi compatibility shim require a model manifest;
- model-loading CLIs require explicit trust acknowledgement;
- fixed-output visualizers require explicit overwrite; and
- MiniGrid opposite headings are no longer immediate neighbors.

The detailed user steps are in
[RR core v2 migration](../migration/rr-core-v2.md). Rollback means reverting the
Phase 1 commits and returning to the old schemas/contracts; new v2 artifacts and
checkpoints are not backward-readable. Existing model archives are not rewritten,
but their generated sidecars and any v2 results may be removed after rollback.

## External compatibility baseline

Implementation decisions were checked against current authoritative sources
before coding. The tested environment is Python 3.11.15 with NumPy 1.26.4,
Gymnasium 1.0.0, Stable-Baselines3 2.4.1, PyTorch 2.2.2, PyYAML 6.0.2,
matplotlib 3.10.0, and pytest 9.1.1.

- Gymnasium's official Taxi documentation was consulted for the 500-state
  encoding and six discrete actions. The installed 1.0.0 implementation was
  used to corroborate the exact index formula; Gymnasium's 404 reachable MDP
  states are not the thesis explanation universe.
- Stable-Baselines3 2.4.1 official source was consulted for deterministic
  `predict`, non-vectorized observation handling, space declarations,
  environment-free `DQN.load`, file-like `BytesIO` loading, and its cloudpickle
  deserialization boundary.
- MiniGrid 3.0.0 official versioned source was consulted for the exact `left`
  and `right` direction updates. Context7 did not expose the step implementation,
  so the installed 3.0.0 behavior and official tagged source are covered by
  all-heading parity tests.
- NumPy's official scalar/array documentation was consulted for integer scalar,
  zero-dimensional, and exact `(1,)` action normalization. Context7 did not
  expose a 1.26.4-specific corpus, so the installed 1.26.4 behavior is covered
  directly by contract tests.
- PyYAML's official safe-loader/dumper documentation was consulted for
  primitive-only serialization and rejection of Python-specific tags.
- pytest's official collection-hook documentation was consulted when removing
  the global collection filter; the installed 9.1.1 collection result is the
  validation authority.
- setuptools' official dynamic-metadata, `src`-layout, and `project.scripts`
  documentation was consulted for AST-first `attr` version resolution, the
  installed `stache` entry point, and wheel smoke test.

Context7 covered current SB3 file-like loading and setuptools dynamic metadata.
Where it lacked the exact installed-version implementation (notably MiniGrid's
turn step), the review used official tagged source plus runtime contract tests.
No external source overrides the thesis' normative scientific definitions.
