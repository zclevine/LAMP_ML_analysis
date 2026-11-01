# Handoff: running the LAMP design portal

For someone standing this up who did not write it — in particular for
integrating it the way PrimerCast is integrated at the Broad.

If you read one thing, read [The constraint that decides your
architecture](#the-constraint-that-decides-your-architecture).

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

## You need the weights, and they are not in git

Deliberately — the portal is public-facing and there is a provisional patent
application. Five checkpoints, 2.8 MB total:

```
tp60_A_v0t9/seed_{0,1,2,3,42}/best_model.pt
```

Ask Sameed. Point `LAMP_DESIGN_WEIGHTS` at the directory that contains the
`seed_*/` folders, or pass `--weights`.

> **Trap:** in the dev checkout those five files are *symlinks* into a training
> run directory. `docker cp`, `modal volume put` and `tar` will all happily copy
> the links instead of the data. Dereference first: `cp -rL`.

## Try it in two minutes

```bash
# GPU or CPU, whichever the box has
LAMP_DESIGN_WEIGHTS=/path/to/tp60_A_v0t9 \
  lamp-design design --in examples/H5N1_HA_demo.fasta --stride 1 --out /tmp/demo
```

`examples/H5N1_HA_demo.fasta` ships with the repo: ten 2025 H5N1 hemagglutinin
genomes from NCBI, 1,715–1,751 nt, taken from PrimerCast's own `target_seqs/`.
They are ragged, so the run exercises the MAFFT alignment path.

In Docker (`--user` matters — the image runs as a non-root user, so without it
every write to `/data` fails with "Permission denied" *after* the pipeline has
done all its work):

```bash
docker build -t lamp-design:cpu .

docker run --rm --user "$(id -u):$(id -g)" \
  -v /path/to/tp60_A_v0t9:/weights:ro \
  -v "$PWD":/data -v "$PWD/examples":/examples:ro \
  lamp-design:cpu \
  lamp-design design --in /examples/H5N1_HA_demo.fasta --stride 60 --out /data/demo
```

For the UI:

```bash
docker build -t lamp-design:portal -f Dockerfile.portal .
docker run --rm -p 8080:8080 -v /path/to/tp60_A_v0t9:/weights:ro lamp-design:portal
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
irrelevant. The GPU lives in our Modal workspace and the weights never leave
it.

This is what the Modal harness exists for. A full sweep is **84 GPU-seconds,
about $0.046**, on one L40S.

You need either a Modal token (workspace-scoped — it is a key to the whole
account, so prefer not to) or, better, the `@modal.fastapi_endpoint` front door
with a rotatable API key, which is on Sameed's list. With that, your image needs
only Streamlit plus `requests`.

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

**Ambiguity codes are rejected, not cleaned.** `N`/`R`/`Y` in an input sequence
is an error with the position named. Silently substituting a base would change
the design without telling you. Six of the 53 sequences in PrimerCast's `H5.fa`
hit this; the demo panel is built from the 47 that do not.

## The portal is slow on a first visit

~21 s on a fresh browser, ~3 s on a repeat. Attribution, the fix, and a
30-second browser test that identifies which cause dominates are in
[`deploy/streamlit/perf/README.md`](../deploy/streamlit/perf/README.md).

Short version: **set `min_instances = 1`** — PrimerCast ships
`cloud_run_min_instances = 0`, so for a low-traffic tool most visits are cold
starts. Everything else (compression, request count, JS parse) is smaller.

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
3. **No dimer/hairpin ΔG filter.** Settled in `ARCHITECTURE.md`, not yet
   implemented. The model never saw non-specific amplification, so this has to
   happen outside it.
4. **Gap columns are warned about, not excluded.** The design notes say primers
   should avoid gapped regions; `prepare` currently only warns above 50%.
5. **Fan-out is not a speedup for one run.** `design_fanout` exists and works,
   but a whole sweep is 84 GPU-seconds, so splitting it leaves each shard less
   work than its own cold start. Use `design_single_container` for a single
   design; fan-out is for many concurrent users.

## Where to look next

| Question | File |
|---|---|
| Why is it built this way? | `docs/ARCHITECTURE.md` |
| Modal setup, tuning, spend limits | `deploy/modal/README.md` |
| Running and testing the UI | `deploy/streamlit/README.md` |
| What the CLI takes | `lamp-design <command> --help` |
