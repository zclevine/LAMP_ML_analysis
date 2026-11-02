# Integrating the LAMP design portal — implementation guide

Audience: whoever (or whatever) wires this into an existing Streamlit-on-Cloud-Run
deployment. Written to be read by a coding agent, so every claim points at a
file and every number is measured, not estimated.

Companion documents: [`../HANDOFF.md`](../HANDOFF.md) for the deployment
decision, [`../ARCHITECTURE.md`](../ARCHITECTURE.md) for why the engine is
built the way it is, [`../../deploy/modal/README.md`](../../deploy/modal/README.md)
for Modal operations.

---

## 1. What you are integrating

A design run is: tile every position of a target with candidate 8-primer LAMP
sets → compute real thermodynamics per (primer, target) pair → score every
candidate against every target with a 5-seed neural ensemble → return the 20
best, spread across loci.

**It is too slow to hold an HTTP request.** 84 GPU-seconds on an L40S for a
full 5 kb sweep; ~4 minutes on 8 CPU cores. So the integration shape is
**submit-and-poll**, not request-response. That is not a preference, it is the
constraint everything else follows from.

| Path | Entry point | Hardware |
|---|---|---|
| Remote GPU (recommended) | `design_single_container` in [`deploy/modal/app.py`](../../deploy/modal/app.py) | 1 GPU + 8 CPU |
| HTTP front door | `http_submit` / `http_status`, same file | thin client calls these |
| Local CPU | `lamp_design.commands.design.design()` | whatever the container has |

---

## 2. The three integration shapes

### Shape A — thin client calling our HTTP endpoints (least for you to own)

Your container needs **Streamlit + `requests`**. No torch, no model weights,
no Modal SDK. We hold the GPU and the weights.

Endpoints are in [`deploy/modal/app.py`](../../deploy/modal/app.py), section
`HTTP front door`, deployed with `LAMP_DESIGN_HTTP=1 modal deploy`:

```
POST /lamp-design-submit     Authorization: Bearer <key>    -> {"call_id": ...}
GET  /lamp-design-status?call_id=...                        -> {"state": ..., "envelope": ...}
```

`state` is one of `queued` / `running` / `succeeded` / `failed`. Poll every
2-3s; a full run is ~30-90s wall clock.

Auth notes that matter for implementation:
- `401` means the key is wrong or absent. **Fails closed** — no configured keys
  means no access (`_authorized`, same file).
- `400` means the request is malformed, and the body says how. Validation runs
  *before* any GPU is spawned, so a bad request costs nothing.
- `429` means our spend ceiling is hit, not that your request was bad. Retry
  later; do not retry immediately.
- Keys are a comma-separated list server-side, so rotation is additive: we add
  yours, you switch, we drop the old one.

### Shape B — your own Modal deployment

```bash
modal deploy deploy/modal/app.py
```

Then Shape A pointed at your own endpoints, or the Modal SDK directly
(`modal.Function.from_name`). You own the bill and the weights upload. See
[`deploy/modal/README.md`](../../deploy/modal/README.md) — the volume upload is
the only fiddly step, and its symlink trap is documented there.

### Shape C — everything in your container, CPU only

`LAMP_DESIGN_BACKEND=local`. Self-contained, no third-party compute.
See §6 for whether it fits your limits.

---

## 3. The request and result contracts

Build a request with [`deploy/streamlit/_backend.py`](../../deploy/streamlit/_backend.py)
`build_design_request()` rather than by hand — it is the single definition of
the shape, and it is what our own UI uses.

```python
{
  "run_id": "20261102-1200-abc123",   # yours; echoed back
  "fasta": ">t1\nACGT...",            # the text, not a path
  "master_mix": "SS",                 # SS | BST | WS
  "target_concs": [10000.0, 100.0],   # the sweep
  "rank_conc": 10000.0,               # which one decides the ranking
  "n_sets": 20,
  "stride": 1,                        # 1 = every position
  "n_geometries": 5,                  # layouts per position
  "min_separation": 30,
  "tp_threshold_minutes": 60.0,
  "pre_aligned": false
}
```

Validation and its limits live in `_validate_request()` in
[`deploy/modal/app.py`](../../deploy/modal/app.py). Caps are enforced
server-side, so you cannot exceed them by asking.

The result envelope is `lamp-design/modal-result@1`:

```python
{
  "schema": "lamp-design/modal-result@1",
  "run_id": ...,
  "timing_seconds": {"prepare": .., "thermo": .., "infer": .., "total": ..},
  "result": {
    "sets": [ {"set_id", "anchor", "geometry_id", "rank", "primers": [...],
               "predictions": [...], ...} ],
    "funnel": {...},        # candidate counts stage by stage
    "warnings": [...],      # surface these; they are actionable
    "params": {...}         # what produced this result
  }
}
```

Render it with [`deploy/streamlit/_results.py`](../../deploy/streamlit/_results.py)
and [`_tables.py`](../../deploy/streamlit/_tables.py) if you want our tables, or
read the dict yourself. `write_outputs()` in
[`src/lamp_design/report.py`](../../src/lamp_design/report.py) produces the same
four files a CLI run does (`.sets.csv`, `.per_target.csv`, `.json`,
`.summary.txt`).

---

## 4. Things that will bite you, in order

### Predicted Tp can be negative — this is correct

The model was trained with `regress_tp_cutoff = -100`; the ground-truth labels
reach −4.611, because measured Tp comes from extrapolating a fitted
amplification curve's threshold crossing back past t=0. So a top-ranked set may
score −5.46.

**Present the rank, not the number.** `format_tp()` in
[`src/lamp_design/report.py`](../../src/lamp_design/report.py) bands anything
under 5 minutes as `<5 min` — 5 is the 25th percentile of all measured Tp, so
that band is "the fastest quartile of everything ever measured" and the model
does not resolve inside it. Do not re-clip at 0: doing so creates ties at the
top of the ranking and we measured it **dropping the model's own best set out
of the returned 20**.

### Ambiguity codes are rejected, not cleaned

`N`/`R`/`Y` in an input is an error naming the position. Silently substituting
would change the design without telling anyone. Surface the message verbatim;
it tells the user what to fix.

### ViennaRNA version is load-bearing

The installed `RNAduplex` decides the binding ΔG every prediction is
conditioned on, and a wrong version **does not crash** — it quietly returns
different numbers. If you change it, run
[`scripts/check_dg_reference.py`](../../scripts/check_dg_reference.py); it
compares six fixed pairs against values measured on 2.4.17 and exits non-zero
on drift. Both our images guard this, by different means — see the comments in
[`Dockerfile`](../../Dockerfile).

### primer3-py must stay below 2.1

2.1.0 changed the santalucia salt correction and shifts every Tm by +2.17 °C
off the values the model was trained on. Pinned in three places; all three must
agree.

### The container runs as non-root

Mount `-v "$PWD":/data --user "$(id -u):$(id -g)"` or every write fails with
"Permission denied" *after* the pipeline has done all its work.

---

## 5. Cloud Run specifics

Our portal images already match the contract PrimerCast uses, so nothing new
is needed: `$PORT` (default 8080), `0.0.0.0`, headless,
`STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`, `EXPOSE 8080`.

Two configuration items that are not optional:

- **`max_instance_request_concurrency` must be > 1.** Streamlit's persistent
  websocket and its file-upload XHRs are separate Cloud Run requests; routing
  the upload to a fresh sessionless instance returns 400. PrimerCast already
  uses 8 with session affinity.
- **`client_max_body_size` / upload cap.** Streamlit buffers an entire upload
  in memory *before* the app sees a byte, so the app's own 5 MB check does not
  protect memory. Set the server cap too and keep them equal.

**First-load time.** Cold start dominates, because scale-to-zero means the
first visit after idle builds the whole container before sending a byte.
Ranked fixes and a browser test that identifies which cause dominates are in
[`deploy/streamlit/perf/README.md`](../../deploy/streamlit/perf/README.md).
Cheapest effective fix: a scheduled ping to `/_stcore/health`.

For Shape A, prefer
[`deploy/streamlit/perf/Dockerfile.thin`](../../deploy/streamlit/perf/Dockerfile.thin) —
it drops torch, ViennaRNA and MAFFT, which are never executed when compute is
remote, and attacks the cold start at its source rather than paying to hide it.

---

## 6. Measured numbers, so you can size things

Full stride-1 run on [`examples/H5N1_HA_demo.fasta`](../../examples/H5N1_HA_demo.fasta)
(10 × ~1.7 kb H5N1 HA, 7,583 candidate sets, 150,228 model evaluations):

| | wall clock | notes |
|---|---|---|
| 1 × L40S (Modal) | **~19 s** internal | the default |
| 1 × L4 | 37 s | |
| 8 CPU cores | **241 s** | peak RSS 3.43 GB |

A full 5 kb × 10 target sweep is **84 GPU-seconds ≈ $0.046** on one L40S.

**Does CPU fit a 600 s Cloud Run timeout?** 241 s on 8 cores extrapolates to
~480 s on 4 — inside, with ~20% margin, and 3.43 GB against a 4 GiB cap. That
is an extrapolation from an 8-core measurement, not a measurement. **Measure
with a real panel before relying on it**, and keep `stride` as the lever: it
scales cost roughly linearly.

**Do not fan out for a single run.** Measured on L40S, warm containers:
single container 28.5 s wall / 18.8 GPU-s against 10-way fan-out 33.1 s wall /
70-79 GPU-s. Fan-out is slower *and* ~4× the cost, because the orchestrator
must ship 7,583 primer sets out and 150,228 predictions back, and that
serialisation exceeds the inference it parallelises. `design_fanout` exists for
concurrent users and oversized inputs, not for latency.

---

## 7. Verifying an integration

```bash
# 1. Environment is capable at all (CPU, ~30s, no GPU billed)
modal run deploy/modal/app.py --mode doctor

# 2. The model actually loads on a GPU
#    (doctor_gpu runs as part of --mode doctor)

# 3. A real design, end to end
modal run deploy/modal/app.py --mode sync \
    --fasta examples/H5N1_HA_demo.fasta --stride 20 --n-sets 5

# 4. The spend ledger (free to call)
modal run deploy/modal/app.py --mode budget
```

`doctor` is the one to run first and the one to run when something is wrong:
it reports tool versions, whether the weights volume resolves, and
`engine_api_missing` — a drift check asserting every engine function the Modal
layer calls still exists. Two of the five bugs found during bring-up were
caught here, before any GPU time was billed.

The test suite is 496 tests; `pytest -q`. Tests needing real weights skip
without `LAMP_DESIGN_WEIGHTS`, so check the skip count rather than trusting a
green tick — `.github/workflows/ci.yml` prints it deliberately.

---

## 8. Known gaps, so you do not go looking for them

1. **No off-target / specificity screening.** Accepted for v1.
2. **No dimer/hairpin ΔG filter.** The model never saw non-specific
   amplification, so this has to happen outside it. Not yet built.
3. **Gap-rich alignment columns are warned about, not excluded** from tiling.
   On the demo panel 1.2% of candidates have a primer over a column that is
   ≥50% gaps.
4. **Design quality is unvalidated.** Ranking is reproducible and the model
   reproduces published accuracy on held-out pathogens; that top-ranked sets
   are *good* sets needs wet-lab confirmation.

None of these block integration. All are in
[`../HANDOFF.md`](../HANDOFF.md) with more detail.
