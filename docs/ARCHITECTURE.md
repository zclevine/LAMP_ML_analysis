# LAMP design engine — architecture

ML-guided LAMP primer-set design. Generates candidate eight-primer sets across a
target region, scores each against every input sequence with the trained
Simplified Lyra ensemble, and returns the top N positionally-diverse sets.

## Pipeline

```
FASTA ──> prepare ──> generation ──> thermo ──> encoding ──> inference ──> report
          (targets)   (candidates)  (features)  (tensors)   (Tp)          (top-N)
```

All stages exchange the dataclasses in `src/lamp_design/types.py`. That module is
the contract: implement against it, don't change it without saying so.

| Stage | Module | Owner |
|---|---|---|
| Input parsing, QC, consensus | `commands/prepare.py` | agent |
| Candidate generation | `generation.py` | agent |
| Thermo features (Tm/GC/mismatch/ΔG) | `thermo.py` | core |
| Sequence + physics encoding | `encoding.py` | core |
| Model loading, ensemble inference | `models/` | core |
| Ranking, diversity, output | `report.py` | agent |
| **Orchestration of all of the above** | `commands/design.py` | core |
| CLI | `cli.py`, `commands/` | agent |
| Deployment scaffolding | `deploy/` | agent |

### One entry point

`lamp_design.design(panel, ...)` runs the whole pipeline and is what the CLI,
Streamlit and Modal all call — the orchestration exists once, not three times.
It records what it did into `DesignResult.params` (stride, geometries,
conditions, device, per-stage timings), so an output file says what produced it.

The names in `lamp_design/__init__.py` are the **public surface** the front ends
integrate against; import `lamp_design.predict_tp`, not
`lamp_design.models.bridge.predict_tp`, so internal layout can move. They load
lazily, so `import lamp_design` does not import torch — `prepare` and `--help`
stay fast. `tests/test_public_api.py` pins that surface.

## Frozen conventions

These match the trained checkpoint and **must not change** without retraining:

- Primer layout order along the target: `F3, F2, FL, F1c, B1c, BL, B2, B3`
- Forward-strand types: `F3, F2, B1c, BL`. The rest are stored reverse-complemented.
- Per-primer physics features: `mismatches, gc_fraction, tm, dg_ViennaRNA`
- Global physics features: `total_mismatches, mean_dg_ViennaRNA`
- Buffer: Na⁺ 50 mM, Mg²⁺ 8 mM, dNTP 1.4 mM, primer 100 nM, 65 °C
- Master mixes: `SS, BST, WS` (3-way one-hot); concentration encoded `log10(conc) - 3`

## The model

Simplified ("Exploratory") Lyra — bidirectional PGC + S4D, `d_model=64`,
`dropout=0.2`, mean-pooled, linear regression head. Five seeds (`0,1,2,3,42`)
ensembled by averaging predictions.

Checkpoint: `LAMP_experiment_20260905_021750_tp60_A_v0t9/seed_*/…/best_model_*.pt`
(config is a field-for-field match with the shipped production
`command_line_args.json`). Each checkpoint embeds:

- `model_state_dict` — weights
- the 34 physics feature names with `mean`/`std` — the physics standardizer
- `label_mean` / `label_std` — to recover predictions in minutes

Weights live **outside** this repo. Resolve via `LAMP_DESIGN_WEIGHTS` (env var)
or `--weights`; never commit them.

Input encoding is 19 channels: 8 ACGT (primer + target) + 1 spacer + 3 master-mix
+ 1 concentration + 6 physics (4 per-primer broadcast + 2 global).

## Design decisions worth knowing

Settled with the team; don't silently revert these.

- **Tile the whole input.** No conserved-region pre-selection — more conserved
  sites should score better on their own, so pre-filtering only adds bias.
- **No Tm window filter.** The 59–61/64–66 °C ranges are PrimerExplorer's and
  exclude essentially all of the training distribution (every LML primer type
  sits near 58 °C). Tm is a model *input*, not a filter.
- **Permissive GC (20–80%)** — matches the LML design space.
- **Deterministic geometries.** Enumerate a small fixed set of spacing/length
  geometries per anchor, including the canonical `(5,0,0,5,0,0,5,0)` spacing at
  length 19–20, which is where the model is best calibrated. No global RNG.
- **Dimer ΔG as a filter.** The model never saw non-specific amplification, so
  homo/heterodimer screening has to happen outside it.
- **Input cap on `n_targets × alignment_length`**, not sequence count.
- **Return 20 sets spread across loci**, not the top 20 by score — otherwise
  you return one design twenty times.
- **No consensus-only pre-screen in v1.** GPU capacity means every candidate is
  scored against every target; no approximation to validate.

## Not implemented

Deliberately listed, because each of these is easy to assume is present:

- **Dimer / hairpin ΔG filtering.** Listed above as a design decision and
  **not written yet**. The model never saw non-specific amplification, so
  nothing in the current pipeline rejects a set whose primers prefer each other
  to the target. `thermo.compute_dg` already does batched duplex ΔG, so the
  machinery exists; the primer-vs-primer screen does not.
- **Specificity / off-target screening.** There is no exclusivity check of any
  kind — not against a host genome, not against related organisms. Every score
  is an *inclusivity* statement about the input panel. PrimerCast's bowtie2
  stage is the obvious model to borrow.
- **`lamp-design evaluate`.** Scoring primer sets a user already has is still a
  stub. It needs no new science — thermo plus inference minus generation — but
  it is what a portal visitor with existing primers will reach for first.
- **Validation of design *quality*.** What is validated is that this code
  reproduces the published model's predictions (`tests/test_parity.py`). That
  the top-ranked designs are good designs is a separate claim needing a
  retrospective check against known-good LAMP sets.
- **Absolute Tp calibration outside the training distribution.** On synthetic
  random templates with zero mismatches the ensemble predicts Tp ≈ 0 min, which
  is not physical; the clip at 0 is doing real work. Treat the *ranking* as the
  product and absolute minutes as indicative until checked on real pathogen
  sequence.

## Performance notes

**The bottleneck has moved twice. Read this before optimizing anything.**

Once ΔG is batched it drops to a few percent of a run, and *model inference*
dominates — not thermodynamics. An early draft of this document said the
opposite; that was true of the reference code and false of this pipeline.

Measured 2026-10-30, 8 CPU cores + one NVIDIA L4, on 1 kb × 10 targets ×
stride 2 × 4 conditions = **74,640 predictions** (one prediction = a
(set, target, condition) triple through all 5 seeds):

| configuration | time | throughput | vs CPU baseline |
|---|---|---|---|
| CPU, no kernel cache | 450 s | 166 pred/s | 1.0× |
| CPU, S4D kernel cache | 363 s | 206 pred/s | 1.24×† |
| **GPU (L4), batch 512** | **30.1 s** | **2,480 pred/s** | **15.0×** |
| GPU (L4), batch 4096 | 32.1 s | 2,323 pred/s | 14.0× |
| GPU (L4), batch 16384 | 32.3 s | 2,308 pred/s | 13.9× |

† contended; the clean interleaved figure for the kernel cache alone is **1.10×**
(see point 3 below). The GPU rows are unaffected — they are 13–15× either way.

Three things follow, and two of them are counterintuitive:

1. **The GPU is worth ~15×**, which is what makes it the primary path rather
   than an optimization.
2. **Bigger GPU batches are slightly worse**, and the L4 sat at **18–29%
   utilization** (370 MiB of 23 GB, 46 W of 72 W) the whole time. The GPU is not
   the constraint — `bridge.encode_pairs` is. It encodes one `(L, 19)` row at a
   time in a Python loop at **5,454 rows/s**, single-threaded, and
   `predict_tp` encodes a whole chunk *before* handing it to the model, so the
   GPU idles through it. That is **4% of a CPU run** but roughly **42% of the
   GPU inference stage**, because going to GPU made inference ~15× faster and
   left encoding untouched.

   The fix is CPU work, and *pipelining* beats parallelizing: overlap the
   encoding of chunk N+1 with the model on chunk N and most of that 42%
   disappears behind work already being done, with one extra thread and no IPC.
   Expect **~1.5× end to end**, not more — 238 s of the 410 s inference stage is
   genuine GPU compute and does not move. A faster GPU buys close to nothing
   until this is done, and not much after.
3. **The kernel cache is real but small — 1.10×.** Measured properly:
   interleaved A/B/A/B on a fixed pre-encoded workload, medians not means
   (5.42 s vs 4.92 s). An earlier sequential measurement said 1.24×, but its
   no-cache arm was competing with a test run; the contended number flattered
   the cache. `S4DKernel.forward` does rebuild the convolution kernel on every
   pass, but the cost is per *batch*, not per row, so at `batch_size=512` the
   FFT convolution already dwarfed it. The cache is kept because it is free and
   provably identical — `max |ΔTp| = 0` — not because it is fast.

   This entry previously predicted the cache would be "substantial". It is not.
   The lesson is in the method: measure interleaved, under no other load.

Numerical agreement across all of the above:

- CPU with cache vs CPU without: **bit-identical**, 100% of rows exact.
- GPU vs CPU: max |ΔTp| = **8.0 × 10⁻⁵ min**, mean 1.3 × 10⁻⁵ — floating-point
  noise from different FFT implementations, on predictions spanning 0–60 min.
  The GPU path is not an approximation.

### The full sweep, measured end to end

The design notes' maximum-size run — 5 kb consensus, 10 targets, **stride 1**,
4 conditions — on one L4, no extrapolation:

| stage | time | share | work |
|---|---|---|---|
| generation | 0.50 s | 0.1% | 23,513 candidate sets |
| thermo | 48.60 s | 10.5% | 235,130 pairs / 1,881,040 windows |
| inference | 410.35 s | **89.0%** | 940,520 predictions @ 2,292/s |
| report | 1.72 s | 0.4% | 20 sets returned |
| **total** | **461 s (7.7 min)** | | peak GPU memory 3.40 GB |

So a worst-case design run is **under eight minutes on a single commodity
GPU**, and 3.4 GB of the L4's 23 GB. The same job on CPU would be ~76 minutes.

Two consequences for deployment: a synchronous web request cannot hold this, so
the portal needs submit-and-poll regardless of backend; and one GPU container
is sufficient — the Modal CPU fan-out is solving the 10.5% slice, which is why
it should be reconsidered rather than scaled.

Caveat visible in that run: the top-ranked set scored Tp = 0.0 min. The panel
was synthetic random sequence with near-zero mismatches, far outside the
training distribution, and the clip at 0 is doing the work. See
"Not implemented" above — ranking is the product, absolute minutes are not yet
trustworthy.

The ΔG fixes below are what moved thermo from dominating to a rounding error.
Measured on the full sweep (5 kb, stride 1, 10 targets): **1,881,040
primer-target slots in 48.6 s**, or 38,700 slots/s. The reference
implementation (`lamp_pipeline/scoring/binding.py`) spawns one RNAduplex
process per (primer, target) pair at 75 pairs/s — the same work would take it
close to **seven hours**. Required fixes:

1. One RNAduplex process for all pairs (or in-process `RNA` bindings), not one each.
2. Dedup ΔG on `(primer_sequence, target_sequence)` — it does **not** depend on
   `primer_type`, so including it inflates the unique set.
3. Tm and GC depend only on the primer, so compute them once per unique primer,
   never per pair.
4. Skip `dg_primer3` and `end_stability_primer3` — the abridged-physics model
   does not consume them.

Mismatch counting must be vectorized (numpy byte comparison), not a per-base
Python loop with set operations.

### ΔG ground truth

Installed RNAduplex is 2.4.17 and the `RNA` Python bindings are 2.7.2, so the
version mismatch had to be checked rather than assumed. It was, against the
`*_dg_ViennaRNA` columns in the training CSVs under
`~/LAMP/lyra/datasets/LAMP/` — the actual values the model learned from, which
is the only ground truth that counts here. On 196 unique pairs from
`LMLv1_corrected_conc.csv`:

| method | throughput | agreement with training data |
|---|---|---|
| one subprocess per pair (reference) | 75 pairs/s | 196/196 exact |
| one batched subprocess | 2,885 pairs/s | 196/196 exact |
| in-process `RNA` bindings | 2,716 pairs/s | 196/196 exact |

So the version difference is a **non-issue for DNA duplexes**, and batching is
worth 38×. Validate any future ΔG change the same way — against those columns,
never against another implementation.

That run also quantified the dedup argument: 400 CSV rows × 8 primers = 3,200
primer-target slots collapse to 196 unique (primer, window) pairs, a 16×
redundancy factor.

## Layout

```
src/lamp_design/
  __init__.py       public surface (lazy, torch-free)
  types.py          contracts — read this first
  generation.py     candidate generation
  thermo.py         Tm / GC / mismatches / dG
  encoding.py       -> (L, 19) tensors
  report.py         ranking, diversity, output
  cli.py            entry point
  commands/
    prepare.py      FASTA -> TargetPanel
    design.py       the whole pipeline, and `design()`
  models/
    lyra/           vendored architecture — keep verbatim
    inference.py    checkpoint loading, 5-seed ensemble
    bridge.py       ThermoTable -> tensors -> Prediction
    fastpath.py     S4D kernel cache (inference-only, output-identical)
  external/         RNAduplex, MAFFT wrappers
  utils/            sequences, params, diagnostics
deploy/modal/       GPU harness (not yet runnable)
deploy/streamlit/   web UI
tests/
```

`models/lyra/` is a verbatim copy of the training-time architecture and its
value is that it stays verbatim — it is what parity is defined against.
Optimizations go in `fastpath.py` and are patched onto module instances, never
edited into the vendored source.

## Conventions

- Python 3.10, `.venv/bin/python` in the repo root has every dependency.
- Type-hint public functions; dataclasses over dicts for anything crossing a
  module boundary.
- `pytest tests/ -v`. Tests must not require GPU or network.
- Each pipeline stage records its surviving-candidate count into
  `DesignResult.funnel` so an empty result can say *which* filter emptied it.
