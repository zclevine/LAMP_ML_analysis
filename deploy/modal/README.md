# Modal deployment

GPU/CPU harness for the LAMP design engine: [`app.py`](app.py).

> **Status: verified end to end on 2026-11-01.** The image builds, `doctor`
> and `doctor_gpu` pass, and `design_single_container`, `design_fanout` and the
> Streamlit backend seam have all produced real results on an L4 — agreeing
> with a local run to within 2.7e-06 minutes. See
> [What has been verified](#what-has-been-verified) for exactly what was
> checked and what still has not been.

## Function topology

| Function | Hardware | Role |
|---|---|---|
| `doctor` | CPU | Reports tool versions, weights volume contents, which engine entrypoints exist. Costs ~nothing. **Run this first.** |
| `doctor_gpu` | 1 GPU | CUDA visibility, checkpoint seed directories, whether the 5-seed ensemble actually loads. |
| `design_single_container` | 1 GPU + 16 CPU | **V1 path.** Whole pipeline in one container, thermo on a multiprocessing pool. |
| `design_fanout` | CPU only | **Scale-up path.** Orchestrator: prepares the panel, shards candidates across `design_shard`, ranks the merge. Never holds a GPU. |
| `design_shard` | 1 GPU + 4 CPU, up to 10 containers | Thermo + inference for one shard of candidate sets. Returns predictions and mismatch counts, not a `ThermoTable`. |

**Why it is split this way — corrected.** An earlier version of this file said
ΔG on CPU was the bottleneck and fanned *thermo* out across 50 CPU containers.
That was wrong. Measured on an L4 over a stride-1 sweep (461.18 s, 940,520
predictions — `docs/ARCHITECTURE.md`):

| stage | seconds | share |
|---|---|---|
| generation | 0.50 | 0.1% |
| thermo | 48.60 | 10.5% |
| **inference** | **410.35** | **89.0%** |
| report | 1.72 | 0.4% |

Batching ΔG (38× on its own, plus a 16× dedup factor) removed thermo as a
rival. So a thermo-only fan-out caps at 1/(1−0.105) = **1.1×**, and the two
functions that implemented it (`thermo_chunk`, `infer`) have been deleted
rather than left deployed and unreachable.

What fans out now is the **candidates**. Each set's predicted Tp is independent
of every other's — ranking is the only step that sees them together, and it
happens in the orchestrator after the merge — so 99.5% of the run is
embarrassingly parallel. Modal bills per GPU-second, so 10 containers cost what
1 does and finish in ~1/10 the wall clock: **~7.7 min → ~1 min**, ~$0.09/run
either way. See [`../README.md`](../README.md) for which path to use when.

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
`LAMP_DESIGN_WEIGHTS=/weights/tp60_A_v0t9`, which is the run directory holding
the five seed subdirectories (`seed_0`, `seed_1`, `seed_2`, `seed_3`,
`seed_42`), each with one `best_model.pt`.

> **`modal volume put` does not follow symlinks.** On the development box
> `~/lamp-weights/tp60_A_v0t9/seed_*/best_model.pt` are all symlinks into the
> experiments directory. Uploading that tree directly puts five broken entries
> on the volume, and nothing complains until the ensemble fails to load inside
> a GPU container. Dereference first:

```bash
PY=/home/ssiddiqu/lamp-design/.venv/bin/python

# -L dereferences. Check you got real files, not links.
mkdir -p ~/lamp-weights-upload
cp -rL ~/lamp-weights/tp60_A_v0t9 ~/lamp-weights-upload/
find ~/lamp-weights-upload -name '*.pt' -type f    # must list five

$PY -m modal volume create lamp-design-weights
$PY -m modal volume put lamp-design-weights ~/lamp-weights-upload/tp60_A_v0t9 tp60_A_v0t9

# Verify: five seed_* directories, each with best_model.pt.
$PY -m modal volume ls lamp-design-weights tp60_A_v0t9
```

The checkpoints total 2.8 MB, so this is a seconds-long upload.

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
    --master-mix SS --target-concs 10000,100 --n-sets 20 --stride 5 --n-geometries 5
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
call = fn.spawn({"fasta": open("targets.fasta").read(), "master_mix": "SS",
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
| `LAMP_DESIGN_MODAL_GPU` | `L4` | GPU type. L4 is the default because every performance number in `docs/ARCHITECTURE.md` was measured on one, so it is the only figure that is not an extrapolation. T4 is 26% cheaper and the GPU only reached 18–29% utilisation, so T4 may be nearly as fast — measure before switching. |
| `LAMP_DESIGN_INFER_CONTAINERS` | `10` | Fan-out width for `design_shard`, and the free plan's concurrent-GPU limit. |
| `LAMP_DESIGN_INFER_SHARD_SETS` | `2000` | Candidate sets per shard. **Measured:** on a 104-candidate input, 5 shards cost 26.6 GPU-s against single-container's 7.3 — each shard pays container start, weights load and torch init. Fan-out is a loss until a shard amortises that, which is what this number is for. |
| `LAMP_DESIGN_THERMO_CHUNK` | `256` | Primer sets per chunk in the in-process thermo pool. A guess, not a measurement. |
| `LAMP_DESIGN_SINGLE_CPUS` | `16` | Cores for the single-container thermo pool. |
| `LAMP_DESIGN_SINGLE_MEMORY_MB` | `32768` | Memory for the single-container path. |
| `LAMP_DESIGN_ALIGN_THREADS` | `4` | MAFFT threads inside prepare. |
| `LAMP_DESIGN_MAX_FASTA_BYTES` | `5242880` | Raw FASTA cap — the crude guard before anything is parsed. |
| `LAMP_DESIGN_MAX_BUDGET` | `50000` | `n_targets × alignment width` cap. **Must track `commands.prepare.DEFAULT_MAX_BUDGET`**; it is passed into `prepare_panel(max_budget=...)`, so prepare is the single enforcement point. |
| `LAMP_DESIGN_MAX_CANDIDATE_SETS` | `200000` | Passed to `generate_candidates(max_candidates=...)`, which thins whole anchors at a uniform stride rather than truncating, so hitting it degrades gracefully. |
| `LAMP_DESIGN_MAX_CONCENTRATIONS` | `4` | Concentrations per request. Each one is another full inference pass over every candidate, so this is the knob that multiplies cost. `deploy/streamlit/_layout.py` mirrors it for its warning; a test keeps the two equal. |
| `LAMP_DESIGN_HOURLY_GPU_HOURS` | `20` | Above this in the trailing hour, **fan-out is withdrawn** and the run proceeds on one container. Currently unreachable — see "Spend circuit breaker" below. |
| `LAMP_DESIGN_DAILY_GPU_HOURS` | `50` | Above this in the trailing 24h, runs are **refused**. |
| `LAMP_DESIGN_TOTAL_GPU_HOURS` | `20` | Above this cumulatively, runs are **refused**. Does not reset on its own. |
| `LAMP_DESIGN_BUDGET` | `1` | `0` disables the ledger entirely, for a deliberate benchmark. |
| `LAMP_DESIGN_BUDGET_DICT` | `lamp-design-budget` | `modal.Dict` holding the ledger. Clear it with `modal dict clear`. |
| `LAMP_DESIGN_WEIGHTS_VOLUME` | `lamp-design-weights` | Volume holding checkpoints. |
| `LAMP_DESIGN_WEIGHTS_SUBDIR` | `tp60_A_v0t9` | Checkpoint directory inside that volume. Matches the local `~/lamp-weights/` layout so one name works both places. |
| `LAMP_DESIGN_RESULTS_VOLUME` | `lamp-design-runs` | Volume holding run artifacts. |
| `LAMP_DESIGN_USE_SECRET` | unset | `1` attaches the `lamp-design-weights` Secret. |

> **Where you set these matters.** Values the *client* evaluates — `gpu=`,
> `cpu=`, `max_containers=` on the decorators — are read in the shell you run
> `modal deploy` from. Everything read inside a function body (shard size,
> chunk size, the budget ceilings) is read from the **container's**
> environment, so those are forwarded into the image via `.env()`. This bit us:
> a run with `LAMP_DESIGN_INFER_SHARD_SETS=25` exported locally still sharded
> at the default 2000 and silently produced one shard. Set them before
> `modal deploy` either way; the forwarding is what makes it take effect.

## Spend circuit breaker

Three ceilings, recorded into a `modal.Dict` after every run and checked
before any GPU work. The hourly one **degrades** (fan-out off, single
container instead); the daily and cumulative ones **refuse**, raising
`BudgetExceeded` — deliberately not a `ValueError`, so a caller can tell "your
request is malformed" from "come back later".

Check usage without spending anything:

```bash
.venv/bin/python -m modal run deploy/modal/app.py --mode budget
```

**The hourly ceiling is currently dormant, twice over.** `max_containers=10`
caps consumption at 10 GPU-hours per wall-clock hour, below the 20 GPU-hour
ceiling; and `LAMP_DESIGN_TOTAL_GPU_HOURS` (20) is not greater than the hourly
ceiling, so any usage big enough to trip the hourly rule has already tripped
the cumulative one, which refuses first. The shipped configuration therefore
refuses rather than degrades. That is the safe direction, but to make graceful
degradation live, raise the total ceiling above the hourly one.

**What this ledger does not see**, because a budget control trusted further
than it works is worse than none:

1. Only what goes through `design_fanout` / `design_single_container`.
   `doctor`, other apps in the workspace, and anything run by hand are
   invisible to it.
2. A container killed mid-run (timeout, OOM, preemption) may never reach its
   record call — so usage is under-counted on exactly the failures that waste
   the most.
3. It is advisory, enforced in this file. **Set Modal's own workspace spend
   limit in the dashboard as well**; that one cannot be bypassed by a bug here.

For scale: the free plan's $30 monthly credit is ~37.5 GPU-hours at the L4
rate, so the *credit* is a tighter constraint than 50 GPU-hours/day could ever
be. A full stride-1 design run is ~7.7 GPU-minutes (~$0.09).


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

## What has been verified

Confirmed by real runs on 2026-11-01, workspace `sameedmsiddiqui`, total cost
$0.015 across 8 runs:

| | result |
|---|---|
| Image build | builds; `apt` found `vienna-rna 2.4.17` and `mafft 7.490` in that base image, so `multiverse` is enabled there |
| `RNAduplex` version assert | passes — 2.4.17, matching the box ΔG was validated against |
| `doctor` | all tools present, weights resolve, `engine_api_missing: []` |
| `doctor_gpu` | `cuda_available: true`, `NVIDIA L4`, 5 seed dirs, **ensemble loaded**, `d_input: 19` |
| torch version skew | **resolved.** The base image's torch 2.4.1+cu121 loads checkpoints written under 2.14.1; `torch.load` default changes in 2.6 did not bite |
| numpy 2.2.6 on torch 2.4.1 | works |
| `design_single_container` | real result, 9 sets from 104 candidates |
| `design_fanout` + `design_shard` | real result, 5 shards merged, **same ranking** as single-container |
| Remote vs local parity | identical set ids in identical rank order, identical primers, predictions within **2.7e-06 min** (float32 noise across two torch/CUDA builds) |
| 1×2 condition sweep | `['SS@10000', 'SS@100']` scored remotely, ranked on the high concentration |
| Streamlit backend seam | `spawn` returns in 0.52 s, poll reports `succeeded`, envelope schema matches, `device: cuda` |
| The `fork` ordering constraint in `_thermo_pool` | held — prepare → thermo pool → inference, no hang |
| Spend ledger | records, reports, and survives a corrupt entry |

### Bugs this found

Each of these built a clean image and failed only at run time, which is the
argument for doing step 4 before trusting anything:

1. **`LOCAL_REPO = Path(__file__).resolve().parents[2]`** — fine locally, but
   in the container the file is `/root/app.py`, whose `parents` is `(/root, /)`.
   `IndexError` at import killed every function during startup.
2. **`einops` was never declared** in `pyproject.toml`. It is a module-level
   import in the vendored S4D kernel, on the production inference path; the dev
   box had it installed by hand, so every local run and test passed.
   `tests/test_packaging.py` now compares declared dependencies against actual
   module-level imports.
3. **`_concat_thermo` assigned `target_windows` after construction**, but
   `ThermoTable.__post_init__` validates its shape — so construction always
   raised first and that line could never have run.
4. **Container-side env vars were never forwarded**, so documented tunables
   silently had no effect remotely.
5. **`params` was never populated** on either Modal path, because
   `build_design_result` does not fill it in — `commands.design.design` does
   that itself. A remote `result.json` could not say what produced it.

### Still not verified

1. **A full-scale run.** Everything above used a 4 × 600 nt panel at stride 20
   (104 candidates). The 7.7-minute / 940,520-prediction figure is from a local
   L4, not from Modal, and `INFER_SHARD_SETS=2000` has never been exercised at
   a size where it is the right number.
2. **Fan-out actually paying off.** Measured on the small input it costs 3.6×
   *more* GPU-seconds than single-container, because each shard pays container
   start, weights load and torch init. The crossover point is unmeasured.
3. **MAFFT on the alignment path.** It is installed in the image (and absent
   from the dev box), but no run has fed it ragged multi-FASTA input.
4. **Retry behaviour.** `design_shard` has `retries=1`; no shard has failed.
5. **A multi-*mix* sweep remotely.** `_validate_request` takes one
   `master_mix`; concentrations fan out, mixes do not.
6. **Polling a still-running call.** Every run so far finished inside the
   first poll interval, so the `TimeoutError`-means-running path is still from
   the documentation rather than observed.
7. **`modal.Dict` contention.** The ledger is a read-modify-write, which is
   not atomic across concurrent shards; with `max_containers=10` a lost update
   would under-count. Only the orchestrator records, so this is currently
   moot — but it would matter if shards ever recorded individually.

### An operational trap worth knowing

After `modal deploy`, a **warm container from the previous version can still
serve calls.** The `params` fix above appeared not to work for one run because
of this: the call landed on a warm container running the old code, which showed
up as a suspiciously fast 7.0 s completion against a 175 s cold start. If a
freshly deployed change seems to have no effect, check whether the run was
warm before debugging the code. `modal run` builds an ephemeral app and always
gets fresh containers, so it is the honest way to test a change.
