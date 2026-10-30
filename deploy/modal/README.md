# Modal deployment

GPU/CPU harness for the LAMP design engine: [`app.py`](app.py).

> **Nothing here has been run.** This repo has no Modal account, so the image
> has never been built, no function has been deployed, and no design has
> executed remotely. The code is written to be correct in shape and runnable
> the moment credentials exist. Steps below are in the order you should do
> them; [What is not verified](#what-is-not-verified) lists exactly what is
> still unknown.

## Function topology

| Function | Hardware | Role |
|---|---|---|
| `doctor` | CPU | Reports tool versions, weights volume contents, which engine entrypoints exist. Costs ~nothing. **Run this first.** |
| `doctor_gpu` | 1 GPU | CUDA visibility, checkpoint seed directories, whether the model loader resolves. |
| `design_single_container` | 1 GPU + 16 CPU | **V1 path.** Whole pipeline in one container, thermodynamics on a multiprocessing pool. |
| `design_fanout` | CPU only | **Scale-up path.** Orchestrator: prepares the panel, fans thermo out, calls `infer`. Never holds a GPU. |
| `thermo_chunk` | CPU, up to 50 containers | One chunk of primer sets × the whole panel → `ThermoTable`. The stage that actually costs time. |
| `infer` | 1 GPU | Assembled features → ensemble inference → ranked `DesignResult`. The only GPU in the fan-out path. |

Why it is split this way: `docs/ARCHITECTURE.md` establishes that **ΔG on CPU
is the bottleneck, not the model on GPU** — the ensemble is ~100K parameters ×
5 seeds, while every unique (primer, target) pair needs a ViennaRNA ΔG. A
single GPU container would therefore idle an accelerator while one core grinds
through RNAduplex. `design_fanout` removes that coupling; `design_single_container`
accepts it in exchange for one cold start and no serialisation. See
[`../README.md`](../README.md) for which to use when.

---

## 1. Install the client

```bash
cd /home/ssiddiqu/lamp-design
.venv/bin/python -m pip install -e '.[modal]'
.venv/bin/python -m modal --version
```

`modal` is an optional extra on purpose — `import lamp_design` must keep
working without it, and it does.

## 2. Create the account and a token

1. Sign up at <https://modal.com> (the free tier includes monthly credits,
   which is more than enough for the `doctor` calls and a few design runs).
2. Authenticate this machine:

   ```bash
   .venv/bin/python -m modal token new          # opens a browser
   ```

   On a headless box, create the token in the web dashboard
   (Settings → API Tokens) and set it instead:

   ```bash
   export MODAL_TOKEN_ID=ak-...
   export MODAL_TOKEN_SECRET=as-...
   ```

3. Confirm:

   ```bash
   .venv/bin/python -m modal token info
   .venv/bin/python -m modal profile current
   ```

## 3. Weights

Weights are **not in this repo and not in the image** — `.gitignore` excludes
`*.pt`/`weights/`, and the image build excludes them again. They live in a
Modal Volume mounted read-only at `/weights`, and the image sets

```
LAMP_DESIGN_WEIGHTS=/weights/LAMP_experiment_20260905_021750_tp60_A_v0t9
```

which is the run directory containing the five seed subdirectories
(`seed_0`, `seed_1`, `seed_2`, `seed_3`, `seed_42`), each with its
`best_model_*.pt`. Upload that directory:

```bash
PY=/home/ssiddiqu/lamp-design/.venv/bin/python
RUN=LAMP_experiment_20260905_021750_tp60_A_v0t9

# The volume is also created automatically on first deploy; this is explicit.
$PY -m modal volume create lamp-design-weights

$PY -m modal volume put lamp-design-weights /local/path/to/$RUN $RUN

# Verify: you should see five seed_* directories.
$PY -m modal volume ls lamp-design-weights
$PY -m modal volume ls lamp-design-weights $RUN
```

The checkpoints are small (~100K parameters × 5 seeds), so this is a seconds-long
upload, not a bulk transfer.

**A different checkpoint without rebuilding the image.** `LAMP_DESIGN_WEIGHTS`
can be overridden by a Secret. Modal has no concept of an optional Secret — a
missing one fails `modal deploy` — so it is opt-in at deploy time:

```bash
$PY -m modal secret create lamp-design-weights \
    LAMP_DESIGN_WEIGHTS=/weights/some_other_run
LAMP_DESIGN_USE_SECRET=1 $PY -m modal deploy deploy/modal/app.py
```

The same mechanism carries a download URL if you would rather fetch weights at
runtime than upload them; the Volume path is the default because it is simpler
and the files are tiny.

## 4. Build the image and check the environment

```bash
cd /home/ssiddiqu/lamp-design
.venv/bin/python -m modal run deploy/modal/app.py
```

With no `--mode` this builds the image and runs `doctor` + `doctor_gpu`. Read
the output before going further:

* `rnaduplex` must equal `rnaduplex_expected` (`2.4.17`). The image build also
  asserts this, so a mismatch should be impossible — but check anyway, because
  a different RNAduplex silently changes ΔG, which is a frozen model input.
* `weights` must be a resolved path, not `UNRESOLVED: ...`.
* `seed_dirs` should list five directories.
* `cuda_available` must be `true`, and `ensemble` must say `loaded` with five
  `ensemble_seeds` — `doctor_gpu` really loads the 5-seed ensemble from the
  volume, so this is the step that proves the weights and the image's torch
  agree.
* `engine_api_missing` lists engine entrypoints that do not exist yet. It
  should now be **empty** — `compute_thermo` and `predict_tp` were the last two
  and both are implemented. If it is non-empty, a design run will fail with a
  `NotImplementedError` naming the missing function rather than silently doing
  something else; nothing here is stubbed. See [`../README.md`](../README.md)
  for the signatures.

The first build takes a few minutes (apt + pip). Later runs reuse the layers.

## 5. Deploy

```bash
.venv/bin/python -m modal deploy deploy/modal/app.py
```

This publishes the app as `lamp-design` so it can be invoked by name from
anywhere — which is how `deploy/streamlit/app.py` reaches it.

## 6. Invoke

**Synchronously, for small jobs** (blocks until done):

```bash
.venv/bin/python -m modal run deploy/modal/app.py \
    --mode sync --fasta targets.fasta \
    --master-mix BST --target-conc 10000 --n-sets 20 --stride 5 --n-geometries 4
```

Input limits, all from the engine's own constants: 1–20 sequences,
`n_targets × alignment width ≤ 50,000`, every sequence ≥ 160 nt, A/C/G/T only
(pass `--pre-aligned` to accept gaps). `--n-geometries` selects a prefix of
`generation.DEFAULT_GEOMETRIES`, which is ordered canonical-first, so any value
includes the canonical `(5,0,0,5,0,0,5,0)` geometry the model is best
calibrated on.

Use `--mode fanout` for the CPU fan-out path.

**Submit and poll**, which is what you want for anything real — design runs
take tens of seconds to minutes and are not interactive requests:

```python
import modal

fn = modal.Function.from_name("lamp-design", "design_single_container")
call = fn.spawn({"fasta": open("targets.fasta").read(), "master_mix": "BST",
                 "target_conc": 10000.0, "n_sets": 20, "stride": 5,
                 "n_geometries": 4})
print(call.object_id)        # the poll token; keep it
```

Later, from any process:

```python
import modal

call = modal.FunctionCall.from_id(token)
try:
    payload = call.get(timeout=0)      # the result payload
except TimeoutError:
    ...                                # still running
```

`.get()` re-raises the remote exception if the run failed, so a rejected input
(too many sequences, over budget, bad master mix) surfaces as the original
`ValueError`. `modal run deploy/modal/app.py --mode spawn --fasta ...` does this
end to end as a demonstration.

The request and envelope shapes are specified once, in
[`../README.md`](../README.md).

## 7. Fetching results later

Modal function outputs expire (roughly a day), so every run also writes to the
`lamp-design-runs` volume: the envelope, plus the engine's own
`report.write_outputs` artifacts, which are identical to a CLI run's.

```bash
.venv/bin/python -m modal volume ls lamp-design-runs
.venv/bin/python -m modal volume ls lamp-design-runs <run_id>
# envelope.json, result.sets.csv, result.per_target.csv, result.json,
# result.summary.txt, plus input.fasta / panel.json / candidates.json
.venv/bin/python -m modal volume get lamp-design-runs <run_id>/ ./
```

That volume accumulates (including each run's input FASTA). Prune it
periodically with `modal volume rm`.

---

## Tuning

Every knob is a deploy-time environment variable, so retuning does not mean
editing code. Set them in the shell you run `modal deploy` from.

| Variable | Default | Meaning |
|---|---|---|
| `LAMP_DESIGN_MODAL_GPU` | `T4` | GPU type. The ensemble is tiny; T4 is already oversized and an A100 would be waste. |
| `LAMP_DESIGN_THERMO_CONTAINERS` | `50` | Fan-out width for `thermo_chunk`. Container concurrency, not input concurrency — thermo is CPU-bound. |
| `LAMP_DESIGN_THERMO_CPUS` | `2` | Cores per thermo container. |
| `LAMP_DESIGN_THERMO_CHUNK` | `256` | Primer sets per chunk. Chunks must do enough work to amortise a cold start; this is a guess, not a measurement. |
| `LAMP_DESIGN_SINGLE_CPUS` | `16` | Cores for the single-container thermo pool. |
| `LAMP_DESIGN_SINGLE_MEMORY_MB` | `32768` | Memory for the single-container path. |
| `LAMP_DESIGN_ALIGN_THREADS` | `4` | MAFFT threads inside prepare. |
| `LAMP_DESIGN_MAX_FASTA_BYTES` | `5242880` | Raw FASTA cap — the crude guard before anything is parsed. |
| `LAMP_DESIGN_MAX_BUDGET` | `50000` | `n_targets × alignment width` cap — the metric ARCHITECTURE.md caps on. **Must track `commands.prepare.DEFAULT_MAX_BUDGET`**; it is passed into `prepare_panel(max_budget=...)`, so prepare is the single enforcement point. |
| `LAMP_DESIGN_MAX_CANDIDATE_SETS` | `200000` | Passed to `generate_candidates(max_candidates=...)`, which thins whole anchors at a uniform stride rather than truncating, so hitting it degrades gracefully. |

Caps are enforced in `_validate_request`, which both entrypoints call before
doing any work, so the two paths are bounded identically. The sequence-count,
budget and minimum-length checks there are estimated from a single pass over
the raw FASTA, against the engine's own `commands.prepare.MAX_SEQUENCES`,
`DEFAULT_MAX_BUDGET` and `DEFAULT_MIN_LENGTH`; `prepare_panel` and
`generate_candidates` then re-check them authoritatively. The pre-parse figures
are lower bounds (MAFFT only widens an alignment with gaps), so passing them
does not excuse the engine's own checks — it just means an absurd upload is
rejected before MAFFT or generation ever runs.
| `LAMP_DESIGN_WEIGHTS_VOLUME` | `lamp-design-weights` | Volume holding checkpoints. |
| `LAMP_DESIGN_WEIGHTS_SUBDIR` | `LAMP_experiment_20260905_021750_tp60_A_v0t9` | Checkpoint directory inside that volume. |
| `LAMP_DESIGN_RESULTS_VOLUME` | `lamp-design-runs` | Volume holding run artifacts. |
| `LAMP_DESIGN_USE_SECRET` | unset | `1` attaches the `lamp-design-weights` Secret. |

Timeouts and retries are set per function in `app.py`: thermo chunks retry
twice (pure function of their inputs, so a retry recomputes the same numbers),
`infer` once, and the two orchestrators **zero** times — retrying an
orchestrator would re-run the entire fan-out.

### A launcher script

The team's existing Modal experiments each ship a `run_*.sh`
(`/home/ssiddiqu/lamp-designer-extracted/lamp-designer-main/experiments/modal/`).
This directory does not, because the deliverable list for it was fixed. The
equivalent preamble, if you want one:

```bash
#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-/home/ssiddiqu/lamp-design/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export LAMP_DESIGN_MODAL_GPU="${LAMP_DESIGN_MODAL_GPU:-T4}"
if ! "$PYTHON" -c "import modal" 2>/dev/null; then
    "$PYTHON" -m pip install -U 'modal>=0.73'
fi
if ! "$PYTHON" -m modal token info >/dev/null 2>&1; then
    echo "Modal is not logged in. Run: $PYTHON -m modal token new"
    echo "or set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET, then rerun."
    exit 2
fi
echo "Modal profile: $("$PYTHON" -m modal profile current)"
"$PYTHON" -m modal run /home/ssiddiqu/lamp-design/deploy/modal/app.py "$@"
```

---

## Pinned tool versions, and why they are pinned hard

The image mirrors the development box exactly:

| Thing | Pin | Source |
|---|---|---|
| `RNAduplex` binary | `2.4.17` (apt `vienna-rna=2.4.17+dfsg-2build2`) | jammy/multiverse |
| `RNA` Python bindings | `2.7.2` (pip `viennarna`) | PyPI |
| MAFFT | `7.490` (apt `mafft=7.490-1`) | jammy/universe |
| torch / CUDA | `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` | the base image the team's other Modal scripts use |

`dg_ViennaRNA` is a frozen model input (`types.PHYSICS_FEATURES`). A deployed
run on a different RNAduplex would produce different predictions from identical
inputs, silently, with nothing crashing. So the image **fails at build time** if
`RNAduplex --version` is not 2.4.17. ARCHITECTURE.md also records that the
2.4.17 binary and the 2.7.2 bindings may not agree with each other, and that the
ground truth is the `*_dg_ViennaRNA` columns in the training CSVs — so whichever
`thermo.py` ends up using, validate against those, not against the other
implementation, and do not bump either pin without redoing that.

If apt later reports "version not found", the Ubuntu archive rotated. Re-pin to
the then-current jammy version and re-validate ΔG before trusting any output.
If apt cannot find `vienna-rna` at all, the base image's `sources.list` is
missing `multiverse`.

---

## What is not verified

Everything in this section needs one real run to settle.

1. **The whole thing on Modal.** No account, so: image never built, nothing
   deployed, no function ever executed. Step 4 is the first thing that will
   tell you whether any of this works.

   What *has* been checked locally, with `modal` stubbed and the real engine
   underneath: 9 of the 11 entrypoints in `ENGINE_API` resolve and are
   callable; `_prepare_stage` runs end to end through the real `read_fasta` →
   `prepare_panel` → `count_candidate_slots` → `generate_candidates`; the
   panel/primer-set JSON wire formats round-trip real objects exactly; chunk
   assembly handles full, partial and empty `ThermoTable`s; and request
   validation rejects every bad input it claims to. See
   [`../README.md`](../README.md).
2. **torch version skew.** The base image carries torch 2.4.1+cu121; the dev
   venv has 2.14.1+cpu. The checkpoints are plain `state_dict`s, which load
   across that gap, but `torch.load` defaults changed in 2.6. `doctor_gpu`
   exists to catch this; if the loader objects, bump `TORCH_IMAGE`.
3. **numpy 2.2.6 on torch 2.4.1.** Pinned to match the dev box. This specific
   pair is untested; if torch complains at import, either bump the base image or
   pin `numpy<2` and accept the drift from local.
4. **apt availability** of both pinned versions inside that base image, and
   that `multiverse` is enabled there.
5. **Chunk size and fan-out width** (`256` sets, `50` containers) are guesses.
   Measure `timing_seconds` from a real run — `prepare` / `thermo` / `infer` are
   reported separately for exactly this purpose — and retune.
6. **Whether the GPU is needed at all.** For a ~100K-parameter ensemble, 16 CPU
   cores may match a T4. If measurement says so, set `gpu=None` on `infer` and
   `design_single_container` and delete the GPU. The fan-out exists because
   thermo dominates, not because the model is heavy.
7. **Polling semantics.** That a not-yet-finished `FunctionCall.get(timeout=0)`
   raises the builtin `TimeoutError` is from the documented behaviour, not
   observed here.
8. **The `fork` ordering constraint** in `_thermo_pool`. The pool forks, and
   forking after CUDA is initialised hangs, so `design_single_container` runs
   prepare → thermo pool → inference and inference is the first thing to touch
   torch. That ordering is correct by construction but has never been run on a
   GPU container.
9. **The two missing engine stages**, `thermo.compute_thermo` and
   `models.predict_tp`. Their names and signatures are this harness's
   expectation, agreed with nobody yet; `doctor` reports them as missing. See
   [`../README.md`](../README.md).
