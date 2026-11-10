# Deploying LAMP-ADAPT

For anyone standing this up who did not write it. Everything below is measured
on the hardware named, not estimated, unless it says otherwise.

Companion documents: [`../SETUP.md`](../SETUP.md) to run it on one machine,
[`ARCHITECTURE.md`](ARCHITECTURE.md) for why the engine is built the way it is.

---

## What this is

An eight-primer LAMP design engine plus two front doors.

Given target sequences, it tiles every position with candidate primer sets,
computes real thermodynamics for each (primer3 Tm, ViennaRNA RNAduplex binding
ΔG), scores every candidate against every target with a 5-seed neural ensemble,
and returns the 20 best sets spread across loci.

| Piece | Where | What it is |
|---|---|---|
| Engine | `src/lamp_design/` | The library + `lamp-design` CLI. No server, no cloud. |
| CLI image | `Dockerfile` → `lamp-design:cpu` | CPU-only. For Snakemake/batch. No server in it. |
| Portal image | `Dockerfile.portal` → `lamp-design:portal` | The Streamlit UI, on `$PORT`, Cloud Run contract. |
| Modal harness | `deploy/modal/` | GPU compute as a service. Deployed, verified. |


## The weights are in the repo — there is nothing to obtain

Five checkpoints, 2.8 MB total, inside the package:

```
src/lamp_design/weights/tp60_A_v0t9/seed_{0,1,2,3,42}/best_model.pt
```

So `pip install -e .` or `docker build` is the whole setup. No download, no
mount, no environment variable, no asking anyone for anything. The Docker build
asserts all five seeds survived packaging, so a build that loses them fails
rather than shipping a broken image.

To run a **different** ensemble — a retrain, an ablation — point
`LAMP_DESIGN_WEIGHTS` at a directory of `seed_*/` folders, or pass `--weights`.
A configured path that does not exist raises rather than silently falling back
to the bundled model, because substituting a different model for the one
someone asked for would change results without telling them.

> **Trap, only if you mount your own:** in a training run directory those `.pt`
> files are often *symlinks*. `docker cp`, `modal volume put` and `tar` will all
> happily copy the links instead of the data. Dereference first: `cp -rL`. The
> bundled copies are already dereferenced.


## Try it in two minutes

```bash
# GPU or CPU, whichever the box has. No flags, no environment: the weights
# are in the package.
lamp-design design --in examples/H5N1_HA_demo.fasta --stride 1 --out /tmp/demo
```

`examples/H5N1_HA_demo.fasta` ships with the repo: ten 2025 H5N1 hemagglutinin
genomes from NCBI, taken from PrimerCast's own `target_seqs/`, MAFFT-aligned
and then trimmed to the largest *contiguous* gapless block — 1,715 of 1,751
columns, 97.9% retained. All ten are the same length with no `-`, which is the
input shape the engine requires, so it doubles as a worked example of
conforming input.

In Docker (`--user` matters — the image runs as a non-root user, so without it
every write to `/data` fails with "Permission denied" *after* the pipeline has
done all its work):

```bash
docker build -t lamp-design:cpu .

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD":/data \
  lamp-design:cpu \
  lamp-design design --in /app/examples/H5N1_HA_demo.fasta --stride 60 --out /data/demo
```

(The demo panel is inside the image at `/app/examples/`, so there is nothing to
mount for it either.)

For the UI:

```bash
docker build -t lamp-design:portal -f Dockerfile.portal .
docker run --rm -p 8080:8080 lamp-design:portal
```

Then open <http://localhost:8080> and pick **Demo panel** on the Target step.


## The constraint that decides your architecture

**PrimerCast's Cloud Run service is 4 CPU, 4 GiB, with a 600-second
per-request timeout** (`terraform/variables.tf`). Its own runs take ~10 s, so
it never comes close.

This workload is not that shape. Measured on the shipped demo panel at
stride 1, same box:

| | CPU (8 cores) | GPU (1 × L4) |
|---|---|---|
| total | **241 s** | **37 s** |
| peak RSS | **3.43 GB** | — |

Same 20 sets in the same order either way — the CPU path is not an
approximation, just slower.

**These numbers improved by 3.9× on 2026-11-01** when input deduplication
landed (identical model inputs are now computed once; see
`docs/ARCHITECTURE.md`). Before it, CPU was 928 s and 4.14 GB, which failed
PrimerCast's limits on both axes at once.

It now **plausibly fits, with thin margin**: 241 s on 8 cores extrapolates to
roughly 480 s on 4 — inside the 600 s timeout, but only 20% clear of it — and
3.43 GB is 80% of the 4 GiB cap. Treat as: possible, not comfortable. A larger
panel, more targets, or a slower vCPU puts it back over. **Measure with your
own panel before committing to a synchronous CPU deployment**, and keep
stride as the lever if it is close.

Three shapes work. Pick by what you control.

### A. Portal on Cloud Run, compute on Modal — recommended

`LAMP_DESIGN_BACKEND=modal`. The container serves only the UI; it submits a job
and polls, so every HTTP request is milliseconds and the 600 s timeout is
irrelevant. The GPU lives in the Modal workspace and the weights never leave
it.

This is what the Modal harness exists for. A full sweep is **84 GPU-seconds,
about $0.046**, on one L40S.

This needs either a Modal token (workspace-scoped — it is effectively a key to
the whole account, so avoid handing one out) or, better, the
`@modal.fastapi_endpoint` front door with a rotatable API key. With the front
door the front-end image needs only Streamlit plus `requests`: no torch, no
ViennaRNA, no checkpoints.

### B. Portal on Cloud Run, compute in the container

`LAMP_DESIGN_BACKEND=local` (the image default), with `--stride` raised until a
run fits your timeout. Stride *n* tiles every *n*-th position, so cost falls
roughly linearly and so does coverage. Entirely self-contained, no third-party
compute, coarser designs. Measure before you pick a stride.

### C. Your own Modal deployment

`modal deploy deploy/modal/app.py` in your own workspace, then shape A pointed
at it. Full control, you own the bill. See `deploy/modal/README.md` — the
volume-upload step and its symlink trap are the only fiddly parts.

> The repo alone is not a working system in any of these: the weights are out
> of git on purpose.


## The request and result contracts

Build a request with [`deploy/streamlit/_backend.py`](../deploy/streamlit/_backend.py)
`build_design_request()` rather than by hand — it is the single definition of
the shape, and it is what the bundled UI uses.

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
[`deploy/modal/app.py`](../deploy/modal/app.py). Caps are enforced
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

Render it with [`deploy/streamlit/_results.py`](../deploy/streamlit/_results.py)
and [`_tables.py`](../deploy/streamlit/_tables.py) if you want our tables, or
read the dict yourself. `write_outputs()` in
[`src/lamp_design/report.py`](../src/lamp_design/report.py) produces the same
four files a CLI run does (`.sets.csv`, `.per_target.csv`, `.json`,
`.summary.txt`).

---


## The optional Fusion model, and where it does and does not exist

Two models ship. **Simplified Lyra** is the default and is reachable
everywhere. **Fusion** -- a four-model blend, see
[ARCHITECTURE.md](ARCHITECTURE.md) -- is reachable only from `evaluate`, and
only where it has been built in.

```bash
lamp-design evaluate --in sets.csv --targets panel.fa --model fusion
```

There is no `--model` flag on `design`, and a test keeps it that way. Fusion
is ~29x slower per prediction, so a stride-1 design run projects to ~1.9 h on
GPU and ~14 days on CPU. The option does not exist there, so it cannot be
started by accident.

### Where it is built in, and where it is not

| image | built from | Fusion |
|---|---|---|
| `lamp-design:cpu` | `Dockerfile` | **yes**, `WITH_FUSION=1` by default |
| `lamp-design:portal` | `Dockerfile.portal` | yes, inherits from the above |
| Modal GPU app | `deploy/modal/app.py` | **no** |
| Modal portal | `deploy/modal/portal.py` | **no** |
| `perf/Dockerfile.thin` | thin client | no, and cannot -- it has no torch |

**Modal builds its own images in Python**, with
`modal.Image.from_registry(...).apt_install(...).pip_install(...)`. It does
not read the Dockerfile. So the Fusion dependencies and the TabPFN checkpoint
baked into `Dockerfile` are absent from the Modal images, and
`--model fusion` is unavailable there until the same three packages and the
same checkpoint are added to those image definitions.

That gap is worth closing deliberately rather than by accident: Fusion on CPU
takes about five hours, so a GPU backend is the only place it is practical,
and Modal is where the GPU is.

### Building without it

```bash
docker build --build-arg WITH_FUSION=0 -t lamp-design:cpu .
```

Fusion takes the CPU image from 3.03 GB to 5.23 GB -- torch plus catboost,
xgboost and tabpfn, plus a 43 MB checkpoint. On Cloud Run that lands directly
on the cold start discussed below, so if you are deploying the portal and do
not need Fusion, build without it.

### The TabPFN checkpoint

Downloaded at **build** time into `TABPFN_MODEL_CACHE_DIR=/opt/tabpfn`, not
committed to the repo. The image therefore needs no network at run time --
verified with `docker run --network none` -- but the build host does need
one. The weights carry an attribution obligation: ship
`third_party/LICENSE-PriorLabs-TabPFN` and display "Built with
PriorLabs-TabPFN".

## Things that will bite you, in order

### Predicted Tp can be negative — this is correct

The model was trained with `regress_tp_cutoff = -100`; the ground-truth labels
reach −4.611, because measured Tp comes from extrapolating a fitted
amplification curve's threshold crossing back past t=0. So a top-ranked set may
score −5.46.

**Present the rank, not the number.** `format_tp()` in
[`src/lamp_design/report.py`](../src/lamp_design/report.py) bands anything
under 5 minutes as `<5 min` — 5 is the 25th percentile of all measured Tp, so
that band is "the fastest quartile of everything ever measured" and the model
does not resolve inside it. Do not re-clip at 0: doing so creates ties at the
top of the ranking and we measured it **dropping the model's own best set out
of the returned 20**.

### Input must be pre-aligned and gapless

Every sequence the same length, no `-`. Ragged input is refused rather than
aligned; a gapped alignment is refused outright. Generation has no concept of
an alignment gap, so a gapped panel would place primers over columns most of
the panel does not cover. Surface the error text — it says what to do.

### Ambiguity codes are rejected, not cleaned

`N`/`R`/`Y` in an input is an error naming the position. Silently substituting
would change the design without telling anyone. Surface the message verbatim;
it tells the user what to fix.

### ViennaRNA version is load-bearing

The installed `RNAduplex` decides the binding ΔG every prediction is
conditioned on, and a wrong version **does not crash** — it quietly returns
different numbers. If you change it, run
[`scripts/check_dg_reference.py`](../scripts/check_dg_reference.py); it
compares six fixed pairs against values measured on 2.4.17 and exits non-zero
on drift. Both our images guard this, by different means — see the comments in
[`Dockerfile`](../Dockerfile).

### primer3-py must stay below 2.1

2.1.0 changed the santalucia salt correction and shifts every Tm by +2.17 °C
off the values the model was trained on. Pinned in three places; all three must
agree.

### The container runs as non-root

Mount `-v "$PWD":/data --user "$(id -u):$(id -g)"` or every write fails with
"Permission denied" *after* the pipeline has done all its work.

---


## Do not change these

**Don't change ViennaRNA without re-running the ΔG check.** The installed
version decides the binding ΔG every prediction is conditioned on, and a wrong
one **does not crash** — it quietly returns different numbers, so you get
different designs with no error anywhere.

The two images guard this differently, for a reason worth knowing:

- The **Modal image** pins the `RNAduplex` binary to **2.4.17** via apt and
  asserts the version at build time.
- The **CLI image** cannot — bioconda has no 2.4.17 build for Python 3.10 — so
  it asserts the property the pin stands for: `scripts/check_dg_reference.py`
  compares six fixed pairs against values measured on 2.4.17 and fails the
  build on any drift.

Measured outcome: the CLI image resolves ViennaRNA **2.7.2**, and it passes the
ΔG check to within 1e-4 kcal/mol. **So the two paths agree**, and that is
verified rather than assumed — which is why the ΔG check is the better guard of
the two. A version string would also pass on a 2.4.17 that had been patched,
rebuilt against different parameters or miscompiled.

If you change the ViennaRNA version, `python scripts/check_dg_reference.py` is
the one thing you must run.

**`primer3-py` is held below 2.1.** 2.1.0 changed the santalucia salt
correction and shifts every Tm by +2.17 °C away from the values the model was
trained on. Pinned in `pyproject.toml`, the Dockerfile and the Modal image —
all three must agree.

**Input must be a pre-aligned, gapless panel.** Every sequence the same
length, no `-` characters. Ragged input is refused rather than aligned, and a
gapped alignment is refused outright: generation has no concept of an
alignment gap, so a gapped panel would silently place primers over columns
most of the panel does not cover. Align and trim to the shared region first.
`examples/H5N1_HA_demo.fasta` is a worked example of the right shape — ten H5N1
genomes aligned and trimmed to a 1,715 nt gapless core.

**Ambiguity codes are rejected, not cleaned.** `N`/`R`/`Y` in an input sequence
is an error with the position named. Silently substituting a base would change
the design without telling you. Six of the 53 sequences in PrimerCast's `H5.fa`
hit this; the demo panel is built from the 47 that do not.


## Cloud Run specifics

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
[`deploy/streamlit/perf/README.md`](../deploy/streamlit/perf/README.md).
Cheapest effective fix: a scheduled ping to `/_stcore/health`.

For Shape A, prefer
[`deploy/streamlit/perf/Dockerfile.thin`](../deploy/streamlit/perf/Dockerfile.thin) —
it drops torch, ViennaRNA and MAFFT, which are never executed when compute is
remote, and attacks the cold start at its source rather than paying to hide it.

---


## The portal is slow on a first visit

~21 s on a fresh browser, ~3 s on a repeat. Attribution, the fix, and a
30-second browser test that identifies which cause dominates are in
[`deploy/streamlit/perf/README.md`](../deploy/streamlit/perf/README.md).

Short version: **set `min_instances = 1`** — PrimerCast ships
`cloud_run_min_instances = 0`, so for a low-traffic tool most visits are cold
starts. Everything else (compression, request count, JS parse) is smaller.


## Measured numbers, so you can size things

Full stride-1 run on [`examples/H5N1_HA_demo.fasta`](../examples/H5N1_HA_demo.fasta)
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


## Verifying an integration

```bash
# 0. The engine, with nothing configured at all. Should just work.
lamp-design design --in examples/H5N1_HA_demo.fasta --stride 60 --top 5 \
    --out /tmp/smoke

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

The test suite is **506 tests, 0 skipped**; `pytest -q`. Everything runs,
including the tests that load real checkpoints and predict — they used to skip
for want of weights and now do not, because the weights are in the package. A
green tick means the model path was actually exercised.

---


## Known issues, in the order they would bite you

1. **Predicted Tp can be negative, and that is correct.** The model was
   trained with `regress_tp_cutoff = -100`, and the ground-truth `tp` labels
   reach −4.611 — negative measured Tp is an artifact of extrapolating a fitted
   amplification curve's threshold crossing back past t=0, not a physical
   claim. So a top set may report e.g. −5.46 min, meaning "faster than
   anything in the training data" (the 0.1st percentile of real measurements
   is 1.76 min). Treat it as a score, not a duration. Whether the UI should
   display it as `<2 min` instead is an open presentation question.

   This was a bug until 2026-11-01: `TP_CLIP_MIN` was `0.0`, which flattened
   in-distribution output into ties and **dropped the model's own top-ranked
   set out of the returned 20 entirely**.
2. **No off-target / specificity screening.** Accepted for v1. A reviewer will
   ask.
3. **No dimer/hairpin ΔG filter, by decision.** The model never saw
   non-specific amplification, so this would have to happen outside it. The
   team chose not to screen for it in v1 — the LML design did not either.
5. **Fan-out is not a speedup for one run.** `design_fanout` exists and works,
   but a whole sweep is 84 GPU-seconds, so splitting it leaves each shard less
   work than its own cold start. Use `design_single_container` for a single
   design; fan-out is for many concurrent users.


## Known gaps, so you do not go looking for them

1. **No off-target / specificity screening.** Accepted for v1.
2. **No dimer/hairpin ΔG filter**, by decision rather than omission. The LML
   design did not screen for it either.
3. **Design quality is unvalidated.** Ranking is reproducible and the model
   reproduces published accuracy on held-out pathogens; that top-ranked sets
   are *good* sets needs wet-lab confirmation.
4. **Fusion is not available on Modal**, because Modal builds its own images
   in Python rather than from the Dockerfile. See the Fusion section above.
5. **Fusion is not exposed in the Streamlit portal.** It is CLI-only today;
   the evaluate page has no model selector.

None of these block integration. All are in
[`ARCHITECTURE.md`](ARCHITECTURE.md) with more detail.
