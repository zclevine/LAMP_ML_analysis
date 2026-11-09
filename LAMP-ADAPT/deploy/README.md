# Deployment

Two deployment surfaces for the LAMP design engine. They are layers, not
alternatives: the UI is the front door, Modal is the compute.

```
                      ┌──────────────────────────┐
  browser  ─────────> │ deploy/streamlit/app.py  │
                      │ upload, validate, render │
                      └───────────┬──────────────┘
                   backend=local  │  backend=modal-*
                 (in-process,     │  (spawn + poll)
                  blocking)       v
                      ┌──────────────────────────┐
                      │ deploy/modal/app.py      │
                      │ CPU thermo + GPU infer   │
                      └───────────┬──────────────┘
                                  v
                         lamp_design.*  (the engine)
```

| | [`modal/`](modal/) | [`streamlit/`](streamlit/) |
|---|---|---|
| What it is | Compute harness: containers, GPUs, fan-out | Web UI: upload, parameters, tables, CSVs |
| Who calls it | The Streamlit app, `modal run`, any Python client | A human in a browser |
| Needs | A Modal account and a populated weights volume | `streamlit`; a Modal deployment only for the remote backends |
| Status | Written, never executed (no account) | Written, never rendered (`streamlit` not installed) |

## Which to use when

**Developing the pipeline, small inputs.** Streamlit with `backend=local`. No
account needed. The run blocks the session, which is fine on a laptop.

**A real design run.** Streamlit with `backend=modal-single`, or
`modal run --mode sync` from a shell. One GPU container plus 16 cores, with
thermodynamics on a multiprocessing pool. **This is the v1 path and probably the
only one you need.**

**Thermodynamics dominating the wall clock.** `backend=modal-fanout`. The CPU
thermo stage fans out over primer-set chunks across many containers and only
`infer` holds a GPU.

### The tradeoff between the two Modal paths

`docs/ARCHITECTURE.md` is explicit that **ΔG on CPU is the bottleneck, not the
model on GPU**: the ensemble is ~100K parameters × 5 seeds, while every unique
(primer, target) pair needs a ViennaRNA ΔG. The trap is a single GPU container
where an expensive accelerator idles while one core grinds through RNAduplex.

`design_single_container` does not dodge that trap, it prices it. One GPU and 16
cores in one container: the cores stay busy through a multiprocessing pool, the
GPU is idle for that stretch, and at design scale that idle time costs cents. In
exchange you get one cold start, nothing crossing the wire, one set of logs, and
a function that is easy to reason about when something breaks.

`design_fanout` dodges it properly. `thermo_chunk` is CPU-only and maps over
chunks at up to 50 containers; `infer` is the only function that ever requests a
GPU, and it runs once over the assembled features. The cost is real complexity:
a shared panel published to a volume, chunk assembly, more failure modes, and a
cold start per chunk that must be amortised by enough work per chunk.

Both paths run the *same* stage sequence (`_prepare_stage` → thermo →
`_infer_stage`); they differ only in how thermo is parallelised. Start with the
single container. Switch when a measured `timing_seconds.thermo` says thermo
dominates — both paths report that breakdown for exactly this reason. And note
the third possibility: for a net this small, 16 CPU cores may match a T4, in
which case the right answer is to drop the GPU rather than to fan out.

## The wire contract

Both backends return the same envelope. The `result` inside it is
**`report.result_to_dict`** — the engine's own JSON-for-the-web-UI
serialisation, which already carries the funnel, the excluded targets, the
warnings and the metric directions. Reusing it means there is no second copy of
the result schema to drift.

### Request

```json
{
  "run_id":               "20261029-084500-1a2b3c4d",
  "fasta":                ">target_1\nACGT...",
  "master_mix":           "BST",
  "target_conc":          10000.0,
  "n_sets":               20,
  "stride":               5,
  "n_geometries":         4,
  "tp_threshold_minutes": 30.0,
  "min_separation":       30,
  "pre_aligned":          false
}
```

`run_id` is optional and generated if absent. Everything is normalised and
bounds-checked server-side by `_validate_request`, against the engine's own
constants:

| Field | Default | Bound |
|---|---|---|
| `master_mix` | `BST` | one of `types.MASTER_MIXES` (`SS`, `BST`, `WS`); `BST` is BstXT-dk |
| `target_conc` | `10000.0` | `> 0`; copies per reaction, encoded `log10(conc) - 3` |
| `n_sets` | `report.DEFAULT_N_SETS` (20) | 1–500 |
| `stride` | 5 | 1–500 |
| `n_geometries` | 4 | 1–`len(generation.DEFAULT_GEOMETRIES)` (5). Selects a prefix, which is ordered canonical-first, so any value includes the canonical `(5,0,0,5,0,0,5,0)` geometry |
| `tp_threshold_minutes` | `report.DEFAULT_TP_THRESHOLD_MINUTES` (30) | — |
| `min_separation` | `report.DEFAULT_MIN_SEPARATION_BP` (30) | 0–100000; 0 degrades to a plain top-N |
| `pre_aligned` | `false` | gaps are accepted only when true |

### Envelope

```json
{
  "schema":  "lamp-design/modal-result@1",
  "run_id":  "20261029-084500-1a2b3c4d",
  "request": { "...": "the request, minus `fasta`" },
  "timing_seconds": {"prepare": 1.2, "thermo": 31.0, "infer": 2.4, "total": 34.6},
  "result":  { "...": "report.result_to_dict(...)" }
}
```

`fasta` is deliberately **not** echoed back: an envelope is persisted and
downloaded, and does not need to carry the user's sequences a second time.

`result` carries `schema_version`, `generated_utc`, `ranking` (the metric
definitions and their directions), `panel`, `funnel`, `excluded_targets`,
`warnings`, `n_sets_returned` and `sets`. Points worth knowing:

* the speed field is named `mean_predicted_tp_minutes_lower_is_faster` and the
  inclusivity field `inclusivity_fraction_targets_below_tp_threshold`. Those
  names spell out their direction because inverting them is, per `report.py`,
  "the single easiest silent bug in the engine".
* `sets[].primers` is a **list** in frozen `types.PRIMER_TYPE_ORDER`
  (`F3, F2, FL, F1c, B1c, BL, B2, B3`), each entry carrying `strand`; `-` means
  the sequence is the reverse complement of its target window.
* `sets` are ordered by `rank`, and that ranking is **spread across loci** by
  `select_diverse` rather than being the top N by score — otherwise one design
  comes back twenty times. A short list is accompanied by a warning; the
  separation constraint is never quietly relaxed to pad it.
* `funnel` is the point of an empty result: it carries each stage's
  surviving-candidate count, so "no candidates" can name the filter that did
  it. The UI shows the first zero-count stage. A result with `sets: []` and no
  funnel is a bug in the engine, and the UI says so rather than showing an
  empty table.

Each run also writes the engine's own artifacts next to the envelope, via
`report.write_outputs`: `result.sets.csv`, `result.per_target.csv`,
`result.json`, `result.summary.txt` — identical to a CLI run's.

## Engine API this scaffolding uses

Neither app reaches into another module's internals; both call public
`lamp_design.*` names, resolved at the call site, failing with the exact name
they wanted. Each entry is (preferred top-level re-export, stage-module path).
Nothing is re-exported from `lamp_design/__init__.py` today, so in practice the
stage-module path always wins; the re-export is checked first so that adding
one later Just Works.

**Nine of eleven resolve against the engine as it stands** (verified by
importing both apps against the real package):

| Entrypoint | Used for |
|---|---|
| `utils.sequences.read_fasta(path)` | one parser for the UI, the CLI and the harness |
| `commands.prepare.prepare_panel(records, *, max_budget, min_length, pre_aligned, force_align, threads, aligner, panel_name)` | QC, exclusions, MAFFT, consensus → `PreparedPanel` |
| `generation.default_geometries()` | the deterministic geometry set, canonical first |
| `generation.count_candidate_slots(template_length, *, geometries, stride)` | the pre-generation funnel entry |
| `generation.generate_candidates(template, *, geometries, stride, length_rules, max_candidates)` | candidates, with the runaway cap |
| `report.build_design_result(predictions, primer_sets, panel, *, ranking_condition, tp_threshold, n, min_separation, mismatches, funnel, excluded_targets, warnings)` | score + rank + diversity-select + funnel bookkeeping |
| `report.result_to_dict(result, *, ranking_condition, tp_threshold)` | the wire payload |
| `report.write_outputs(result, out_prefix, ...)` | the four per-run artifact files |
| `models.inference.resolve_weights_dir(explicit=None)` | `LAMP_DESIGN_WEIGHTS` → checkpoint dir |

**Both former gaps are now closed** (they were `thermo.compute_thermo` and the
`ThermoTable → Prediction` bridge):

| name | what it is |
|---|---|
| `thermo.compute_thermo(primer_sets, panel, *, threads, resolver)` | ΔG/Tm/GC/mismatches, batched and deduped |
| `predict_tp(thermo, primer_sets, conditions, *, weights, device, batch_size, pair_chunk)` | encode + ensemble inference → `Prediction` list |

Note `predict_tp` takes the **primer sets**, not the panel: `ThermoTable` holds
the resolved target windows but not the primer sequences, and encoding a row
needs both. `conditions` is a sequence, so one call covers the whole master-mix
× concentration sweep.

Run `modal run deploy/modal/app.py` once credentials exist: `doctor` reports
`engine_api_present` / `engine_api_missing`, which is the live answer.

### Resolved: both apps now make one call

This section used to record an open decision — the CLI's `design` surface had
nowhere to put master mix, concentration, stride or geometry count, so both
deploy apps re-implemented the stage sequence. Both halves are now done:

* `lamp_design.design(panel, ...)` is the orchestration entrypoint, and it
  records the run's settings and per-stage timings into `DesignResult.params`.
* `lamp-design design` grew `--master-mix`, `--conc`, `--rank-master-mix`,
  `--rank-conc`, `--stride`, `--random-geometries`, `--device`, `--batch-size`,
  `--pair-chunk`, `--max-candidates`, `--min-separation`, `--tp-threshold`.

The Streamlit `local` backend calls `design()` once. **Do not reintroduce a
hand-assembled stage sequence in a deploy app** — that is how the UI and the
CLI drift apart.

## What has and has not run

Both surfaces are scaffolding: the user has no Modal account, so no image has
been built, no function deployed, no design executed, no page rendered.

What *has* been exercised, with `modal` and `streamlit` stubbed and the real
engine underneath:

* 9 of 11 engine entrypoints resolve and are callable; the missing set is
  exactly the two gaps above, and both fail with the name they wanted.
* `_prepare_stage` driven end to end through the real `read_fasta` →
  `prepare_panel` → `count_candidate_slots` → `generate_candidates`, producing a
  real panel, 25 real candidate sets and the real prepare funnel.
* The panel and primer-set JSON wire formats round-trip real `TargetPanel` /
  `PrimerSet` objects exactly, including spans and sequences.
* Thermo chunk assembly over full, partially-populated and empty `ThermoTable`s.
* Request validation rejecting every bad input it claims to, against the
  engine's own `MAX_SEQUENCES`, `DEFAULT_MAX_BUDGET` and `DEFAULT_MIN_LENGTH`.
* FASTA pre-flight validation agreeing with the engine (notably that `N` and
  other IUPAC codes are rejected, not cleaned).
* Run directories distinct and `0700` per run; run ids unique over 300 draws;
  no `chdir()` call in either app (AST-checked).
* All four display tables and their CSVs built from a **real**
  `report.result_to_dict` payload, and both renderers run without raising —
  including the empty-result path, which names `candidates_generated` as the
  stage that emptied the funnel.

`import lamp_design` still works without `modal` or `streamlit` installed, and
nothing under `src/` imports `deploy/`.

Per-surface "what is not verified" lists: [`modal/README.md`](modal/README.md),
[`streamlit/README.md`](streamlit/README.md).
