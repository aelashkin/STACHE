# RR core v2 migration

RR core v2 makes scientific identity, partial-result evidence, continuation
integrity, and model trust explicit. The stricter contracts intentionally reject
ambiguous v1 inputs instead of guessing.

## Compatibility summary

| Surface | v2 requirement | Migration |
| --- | --- | --- |
| RR artifacts | `stache.rr-result` schema 2 and core schema 2 | Recompute from the trusted policy and connector; do not relabel v1 YAML |
| Continuations | `stache-rr-continuation-v2` in-memory checkpoint | Restart v1 searches; v1 payloads cannot resume |
| Model policies | Exact archive fingerprint plus observation/action manifest | Supply `model.manifest.yaml` and explicitly acknowledge trusted archives |
| Legacy experiment loader | Explicit trust; optional connector-bound manifest | Pass `acknowledge_trusted_model=True` and `model_connector=...` when semantic validation is required |
| Custom action oracles | Pure, exact `policy_query_cost(state)` | Implement preflight cost and preserve cumulative counters |
| Legacy Taxi API | Explicit verified `ModelManifest` for model-backed calls | Pass `model_manifest=` or attach `stache_model_manifest` to the model |
| Visualizers | Trusted-model acknowledgement and explicit overwrite | Add the flags shown below |
| MiniGrid headings | Centralized one-turn left/right validation | No topology migration; historical generators already excluded 180° edges |

The Python package version is now single-sourced as `1.0.0` in both runtime and
built-wheel metadata.

## Recompute v1 artifacts

Artifact v1 lacks several identities and independent completeness fields needed
to validate a scientific result. There is no metadata-only converter. Retain a
v1 file as historical evidence if needed, then recompute a new output from the
trusted policy source:

```bash
stache compute-rr \
    --domain taxi \
    --state-universe taxi-factored-500 \
    --seed 2 \
    --model path/to/model.zip \
    --acknowledge-trusted-model \
    --minimum-basis formal_global \
    --counterfactuals both \
    --extent exact \
    --output result-v2.yaml
```

Do not copy v2 schema numbers into an old document. The v2 loader recomputes
canonical identities and completeness evidence and will reject contradictions.

## Restart v1 continuations

Continuation v2 type-tags container kinds and revalidates all checkpoint state,
key, ordering, frontier, formal-layer, cache, and counter fields. A v1 checkpoint
does not carry sufficient evidence and must be restarted from its original seed
and semantic options. Resource ceilings may be raised only when resuming a valid
v2 in-memory continuation.

Serialized result artifacts contain non-resumable continuation metadata only;
they are not a checkpoint transport format.

## Add a model sidecar

New Taxi training runs write the sidecar automatically. For an existing archive,
first independently verify that it was trained with the exact Taxi 500-way
float32 one-hot observation and six-action contract. Only then create the
connector-owned binding:

```python
from pathlib import Path

from stache.explainability.connectors.taxi import TaxiConnector
from stache.explainability.model_manifest import write_connector_model_manifest

write_connector_model_manifest(
    Path("path/to/model.zip"),
    TaxiConnector(),
)
```

The writer refuses symlinked model inputs and existing sidecars by default. Use
`overwrite=True` only after intentionally replacing or re-verifying the model.
The sidecar is a semantic identity record, not proof that an archive is safe or
authentic.

The legacy experiment utility now fails closed unless trust is explicit:

```python
from stache.explainability.connectors.taxi import TaxiConnector
from stache.utils.experiment_io import load_experiment

model, config = load_experiment(
    experiment_dir,
    acknowledge_trusted_model=True,
    model_connector=TaxiConnector(),
)
```

Omit `model_connector` only for a legacy domain that has no registered connector;
the archive is still snapshotted and the trust acknowledgement remains required.

Untrusted CLI/config YAML is limited to 1 MiB and RR result YAML to 16 MiB.
Inputs must be regular non-symlink UTF-8 files. Oversized legacy inputs should be
reviewed and reduced rather than bypassing the parser boundary.

All model-loading commands require a separate trust decision:

```bash
stache compute-rr ... --model path/to/model.zip --acknowledge-trusted-model
stache-viz-rr-taxi ... --acknowledge-trusted-model
stache-viz-policy-map ... --acknowledge-trusted-model
```

## Update custom action oracles

Every oracle must report the exact uncached cost of its next action request
without changing observable state:

```python
def policy_query_cost(self, state) -> int:
    return 0 if self.has_cached_action(state) else 1
```

The value must match the subsequent `policy_queries` counter delta. The core
preflights this cost before calling `action` and rejects mismatches. An uncached
seed requires a positive budget; a zero query ceiling is valid only when the
seed is already cached.

## Update model-backed Taxi compatibility calls

`compute_rr_taxi` still delegates to the generic core and returns its deprecated
legacy mapping, but model semantics are no longer inferred:

```python
result = compute_rr_taxi(
    seed,
    model,
    env,
    model_manifest=verified_manifest,
)
```

Prefer `TaxiConnector`, `ModelActionOracle`, and `compute_rr` for new code.

## Update visualization workflows

The RR visualizer now writes the canonical v2 artifact through the connector
codec. It refuses an existing result unless `--overwrite` is provided. The
policy-map visualizer uses a timestamped directory and refuses an existing fixed
timestamp unless `--overwrite` is provided. Neither visualizer re-queries
scientific fields already stored in an RR result.

## MiniGrid direction verification

The historical Empty and Fetch neighbor generators already used one left or
right 90° turn and excluded the opposite heading. This release centralizes that
rule, validates all four headings against MiniGrid 3.0.0 runtime behavior, and
adds regression tests; it does not change the valid-state topology and does not
require recomputation solely for direction adjacency. MiniGrid is still not
registered with the generic core, and this verification does not decide its
broader object universe, observation codec, state injection, or metric
certificate.

## Rollback

Rollback requires reverting the v2 implementation and returning consumers to
their old schemas and APIs. V2 artifacts and continuations are not readable by
v1 code. Existing model archives are unchanged; generated sidecars and v2 result
files can be retained as inert evidence or removed after the rollback decision.
