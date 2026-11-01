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
| Scoring existing sets (+ metrics) | `commands/evaluate.py` | core |
| CLI | `cli.py`, `commands/` | agent |
| Deployment scaffolding | `deploy/` | agent |

### One entry point

`lamp_design.design(panel, ...)` runs the whole pipeline and is what the CLI,
Streamlit and Modal all call — the orchestration exists once, not three times.
It records what it did into `DesignResult.params` (stride, geometries,
conditions, device, per-stage timings), so an output file says what produced it.

`lamp_design.evaluate(rows, ...)` is the second entry point, for sets that
already exist. It skips generation and takes its target windows either from the
input's `<type>_target_sequence` columns or by locating each primer in a target
FASTA, then shares `thermo` and `predict_tp` with `design` — which is what
makes the two commands' numbers comparable.
`tests/test_evaluate.py::test_evaluate_and_design_agree_on_the_same_primer_set`
pins that: the same set scored both ways must agree to 1e-4.

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
- **Canonical plus sampled geometries, five per anchor.** See
  [Geometry sampling](#geometry-sampling) — this one was changed, because an
  earlier version of this list said "deterministic geometries, a small fixed
  set" and that was not what the design notes asked for.
- **Dimer ΔG as a filter.** The model never saw non-specific amplification, so
  homo/heterodimer screening has to happen outside it.
- **Input cap on `n_targets × alignment_length`**, not sequence count.
- **Return 20 sets spread across loci**, not the top 20 by score — otherwise
  you return one design twenty times.
- **No consensus-only pre-screen in v1.** GPU capacity means every candidate is
  scored against every target; no approximation to validate.
- **`pad_mode="group_by_length"` is the default.** See below; this one was
  revisited with measurements rather than settled by argument.

### Padding: why not reproduce the training `collate_fn`

The training `collate_fn` zero-pads each batch to its longest member, and
`Lyra.forward` mean-pools over the result **without a mask**. So a sequence's
prediction depends on which other sequences happened to share its batch — by
up to 3 minutes. The reasonable-sounding objection is that the model was
*trained* that way, so inference should match it.

It should not, for two reasons.

**There is no single padding to reproduce.** Batches were reshuffled every
epoch, so each sequence was seen with a different, random amount of padding.
The dilution was noise the model trained through, not a feature it relies on.
`group_by_length` pads nothing, which is exactly what the longest member of
any training batch saw.

**It would make the output depend on batch bookkeeping.** A full sweep is
940,520 predictions chunked 20,000 pairs at a time. Under `batch_max` the
returned top-20 would shift with `--pair-chunk` and `--batch-size`, so two
runs of the same input could disagree about the winner. That is disqualifying
for a portal regardless of accuracy.

And the accuracy cost is nil. Both modes scored against measured Tp:

| dataset | distinct lengths | Δρ (amplified) | ΔAUC | mean per-row Δ |
|---|---|---|---|---|
| Borrelia | 1 | 0.0000 | 0.0000 | 0.00 min |
| LASV | 3 | 0.0003 | 0.0006 | 0.11 min |
| mpox | 14 | 0.0017 | 0.0003 | 0.75 min |

The differences are 3rd–4th decimal and **mixed in direction** — `batch_max`
is marginally better on two mpox metrics and marginally worse on a third.
That is noise, not signal. Note Borrelia is bit-identical: all 45 of its sets
encode to the same length, so there is no padding to differ about, which is
also the common case within a single `design` geometry.

`batch_max` is kept, because reproducing a *specific* published number needs
the batching that produced it. It is not the default.

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
- **Validation of design *quality*.** What is validated is that this code
  reproduces the published model's predictions (`tests/test_parity.py`) and
  agrees with measured Tp on three real pathogens (below). That the
  *top-ranked* designs are good designs is a separate claim, needing a
  retrospective check against known-good LAMP sets.
- **Absolute Tp calibration.** Measured on real data and it is real: see the
  bias column below. Treat the *ranking* as the product and absolute minutes
  as indicative.

## Validated against measured Tp

`lamp-design evaluate` scores primer sets that already exist, so the three
pathogen datasets under `~/LAMP/lyra/datasets/LAMP/final/` — which carry
measured Tp — are a direct check on the whole pipeline. Run
2026-10-30 on an L4, ranking under each row's own master mix and
concentration:

| dataset | rows | ρ (amplified) | ρ (censored) | ROC-AUC | bias |
|---|---|---|---|---|---|
| LASV | 25,947 | 0.542 | 0.623 | **0.875** | +5.9 min |
| mpox | 2,088 | 0.394 | 0.603 | **0.874** | −0.4 min |
| Borrelia | 3,240 | 0.525 | 0.829 | 0.984 | +18.0 min |

The paper's headline ROC-AUC is 0.875 (95% CI 0.866–0.883). LASV reproduces it
to three decimals and mpox to 0.874, which is the strongest evidence available
that this reimplementation is the published model and not merely something
correlated with it.

Three things to read off this table rather than from the single number:

1. **Discrimination is much better than regression.** ROC-AUC asks "will this
   amplify inside 30 min" and gets 0.87–0.98. Spearman restricted to reactions
   that *did* amplify asks "how fast, among the ones that work" and gets
   0.39–0.54. Most of the censored ρ is the easy amplify/no-amplify split. The
   portal's job is the first question, which is the one the model is good at.
2. **Absolute minutes carry a real offset**, and its sign is not stable across
   pathogens: +18 min on Borrelia, −0.4 on mpox. This is the concrete version
   of the calibration caveat above. Rank, do not quote.
3. **Censoring convention matters**, so both populations are always reported.
   A blank Tp means the reaction never amplified, censored at 60 min to match
   the training data's `tp60_censored` flag.

### The Tm bug this check found

The same comparison against the training CSVs' own `*_tm` columns showed our
Tm was **+2.17 °C off on every primer**, while mismatches, GC and ΔG were
exact. Cause: **primer3-py 2.1.0 changed the santalucia salt correction.** The
training data was generated with 2.0.x; `>=2.0` with no upper bound resolved to
2.4.0 here. Versions 1.1.0–2.0.3 reproduce the training column exactly,
2.1.0+ do not.

`pyproject.toml` and the `Dockerfile` now pin `primer3-py>=2.0,<2.1`, and
`tests/test_thermo.py` asserts a known Tm value from the training data so a
future bump fails loudly instead of biasing every prediction.

Isolating the effect (same rows, our Tm vs the CSV's) put it at **1.4 min mean
and 5.4 min max per prediction, with the two prediction vectors rank-correlated
at 0.99**. So it mattered for absolute Tp and barely at all for ranking — worth
fixing for exactness, but *not* the cause of the bias in the table above. That
bias is the model generalizing to a held-out pathogen.

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
   utilization** (370 MiB of 23 GB, 46 W of 72 W) the whole time. Part of the
   reason is `bridge.encode_pairs`: it encodes one `(L, 19)` row at a time in a
   Python loop at **5,454 rows/s**, single-threaded, and `predict_tp` encodes a
   whole chunk *before* handing it to the model, so the GPU idles through it.
   That is **4% of a CPU run** but roughly **42% of the GPU inference stage**,
   because going to GPU made inference ~15× faster and left encoding untouched.

   **Correction (2026-11-01).** This section used to conclude from that "the
   GPU is not the constraint", and `deploy/modal/app.py` used to conclude from
   it that a bigger GPU "buys almost nothing". A head-to-head measurement says
   otherwise: on the same input an L40S runs the inference stage **2.71×**
   faster than an L4 (see [The full sweep](#the-full-sweep-measured-end-to-end)).
   Low utilization did not mean idle capacity going to waste — the stage is
   latency-bound on a long sequence of small kernels, and the L40S retires each
   one faster. Utilization percentage was simply the wrong instrument for the
   question "would a faster GPU help"; a stopwatch was the right one.

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

### CPU vs GPU on the shipped demo panel, and what it means for Cloud Run

Measured 2026-11-01 on `examples/H5N1_HA_demo.fasta` (10 × ~1.7 kb H5N1 HA,
1,751 nt consensus), stride 1, 7,583 candidates, 150,228 predictions, same box:

| stage | CPU (8 cores) | GPU (1 × L4) | speedup |
|---|---|---|---|
| generation | 0.22 s | 0.21 s | — |
| thermo | 10.62 s | 10.54 s | — |
| **inference** | **915.64 s** | **64.02 s** | **14.3×** |
| report | 0.38 s | 0.36 s | — |
| **total** | **928.5 s (15.5 min)** | **77.1 s** | **12.0×** |
| peak RSS | **4.14 GB** | — | |

Identical results: the same 20 sets in the same order, max |ΔTp| **5.8 × 10⁻⁴
min**. Thermo does not move because it is CPU work on both paths; inference is
the entire difference, and on CPU it is **98.6%** of the run.

**Superseded by deduplication.** The table above predates input dedup. Re-run
on 2026-11-01 with it:

| | CPU (8 cores) | GPU (1 × L4) |
|---|---|---|
| total | **240.6 s** (was 928.5) | **37.4 s** (was 77.1) |
| speedup from dedup | **3.86×** | 2.06× |
| peak RSS | **3.43 GB** (was 4.14) | — |

CPU gains more than GPU because inference was 98.6% of a CPU run, so removing
65% of it removes most of the whole thing.

**What that does to the Cloud Run question.** PrimerCast's service is 4 CPU,
4 GiB, 600 s per-request timeout (`terraform/variables.tf`), sized for its own
~10 s runs. Before dedup a stride-1 CPU run failed both limits at once: 31 min
against a 600 s timeout, and 96% of the memory cap. After it, 240.6 s on 8
cores extrapolates to **~480 s on 4** — inside the timeout with ~20% margin —
and 3.43 GB is 80% of the cap.

So a synchronous CPU deployment is no longer categorically out, but the margin
is thin and the 4-core figure is an extrapolation from an 8-core measurement,
not a measurement. A larger panel, more targets or a slower vCPU puts it back
over. Modal remains the right default; CPU is now a viable fallback rather
than an impossible one. Both shapes are written up in [HANDOFF.md](HANDOFF.md).

### Geometry sampling

At every anchor the generator lays down the **canonical** layout plus four
drawn per anchor. That is the design notes' scheme, quoted from them:

> "for every sampled position, I think we have the compute to sample multiple
> primersets for that position with different spacings and lengths (i.e. let's
> say 5). Because the LML dataset had gaps of (5, 0, 0, 5, 0, 0, 5, 0), I think
> it would be helpful to as a rule have one of those 5 primerset spacings"
>
> "for our 'default' primerset at that position ... I'd nominate our median
> primerlengths in the LML datasets, which is 19-20."

So `CANONICAL_LENGTHS` is **measured, not chosen**: the per-type median over
LMLv1 + LMLv2 (119,928 sets) is F3 19, F2 20, FL 20, F1c 20, B1c 19, BL 19,
B2 20, B3 20, pooled mean 19.49. The other four draw lengths inside
`LENGTH_BOUNDS` and gaps inside `GAP_BOUNDS` from a normal centred on each
range's midpoint with σ = width/6 — the notes' "mean at the midpoint of
min/max, ~99.7% of draws within range". Draws are clipped rather than
rejected, so the bounds are hard and the number of RNG draws stays independent
of the data.

**Why this replaced a fixed panel.** The repo previously shipped five fixed
geometries, four of which were an implementer's choice rather than a
requirement. A fixed panel explores exactly five spacings however many
positions are tiled; per-anchor sampling explores one per candidate. Measured
on a 3 kb template at stride 1: **18,116 distinct layouts against 5**, at the
same candidate count and the same inference cost. The diversity is free.

**Reproducibility is kept, not traded away.** The notes flag the cost of
sampling themselves — "candidate coverage depends partly on random seed, not
just the length bounds" — so the seed is derived from the input
(`derive_geometry_seed`, sha256 of stride + template). The same FASTA always
returns the same sets, while each anchor still gets its own draw;
`--geometry-seed` overrides. Everything draws from a local
`np.random.default_rng`, never the global numpy RNG, so concurrent design runs
in one process cannot perturb each other — which the reference implementation
did.

Anchors stay uniformly spaced either way: the anchor *is* the F3 start, so a
long draw extends the set rightward instead of displacing it. Only the span
varies, which is why `count_candidate_slots` is exact for fixed geometries and
an estimate for sampled ones.

`DEFAULT_GEOMETRIES` survives as an opt-in fixed panel (`--fixed-geometries`),
because a reproducible candidate count is the right instrument for a
controlled comparison — the L4-vs-L40S benchmark below used it.

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

#### The same sweep on Modal, and on a bigger GPU

Re-measured 2026-11-01 **at the shipped 1×2 condition default** rather than 4
conditions, on a seeded 10 × 5 kb panel (22,773 candidate sets), both runs via
`design_single_container` on Modal:

| stage | L4 | L40S | speedup |
|---|---|---|---|
| prepare | 0.55 s | 0.25 s | 2.2× |
| thermo | 11.70 s | 6.83 s | 1.7× |
| inference | 207.80 s | 76.74 s | **2.71×** |
| **total GPU-seconds** | **220.34** | **84.07** | **2.62×** |
| wall clock (incl. cold start) | 259 s | **95 s** | 2.7× |
| cost at modal.com/pricing | $0.0489 | **$0.0456** | — |

**The L40S is 2.6× faster and marginally cheaper per run**, because its 2.44×
rate ($0.000542/s vs $0.000222/s) is less than its speedup. It is now the
default in `deploy/modal/app.py`. Results are bit-identical between the two:
the same 20 sets in the same order, max |diff| **0.0** across every reported
score — so this is a pure latency win, not a precision tradeoff.

Two other things fell out of that run:

- **Halving the conditions halved inference**, as predicted: 410.35 s at 4
  conditions → 207.80 s at 2, on comparable candidate counts. Thermo is
  condition-independent (it is per primer-target window) and did not move for
  that reason.
- **Thermo on Modal was 4× faster than the local measurement** (11.70 s vs
  48.60 s) on a comparable panel. `design_single_container` spreads thermo over
  a multiprocessing pool and the local CLI path may not; this is an unexplained
  discrepancy, flagged rather than resolved. It does not affect ranking, and
  thermo is now 5% of a run rather than 10.5%.

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

This was later re-checked at much larger scale through `evaluate`, against all
four physics features on all three pathogen datasets — 32,752 primer-window
pairs on LASV alone. Every feature agrees to `max|diff| = 0.00000`; the
residual at the 1e-6 level is float32 storage, not arithmetic. That check is
the one that caught the Tm pin, so it is worth re-running after any dependency
change: `scratchpad/thermo_parity.py` in the session notes, or just compare a
`thermo_from_windows` call against the CSV's own columns.

### Scaling out on Modal

Modal bills per GPU-second, so N containers that each do 1/N of the work cost
the same as one container and finish in roughly 1/N of the wall clock — minus
one cold start each. That makes fanning out close to free, and it is why
`design_fanout` shards **candidate sets** across GPU containers rather than
splitting the pipeline by stage.

Sharding by candidate works because a candidate's predicted Tp is independent
of every other candidate's; ranking is the only step that needs them together,
and it runs on the orchestrator after the merge. An earlier version of that
function fanned out *thermo only* and kept inference on a single GPU, which
caps at about 1.1× no matter how many containers it gets — thermo is 10.5% of
a run.

Free-plan limits (confirmed on modal.com/pricing, 2026-10-30): **10 concurrent
GPUs, 100 containers, $30/month of credit**. A full stride-1 sweep measures
**84 GPU-seconds ≈ $0.046** on one L40S, so the free credit is on the order of
650 full-sweep runs per month.

**Fan-out is the wrong lever for a single run, and the measurement says so.**
The paragraph above is still true about billing, but it omits the constant: a
whole sweep is now 84 GPU-seconds, so splitting it 10 ways leaves ~8 seconds of
work per shard against a cold start plus five checkpoint loads — measured once
at 26.6 GPU-s across 5 shards to do 7.3 GPU-s of work. One container at ~95 s
wall clock is both faster and cheaper than fanning that out.

`design_fanout` keeps its place for the two cases where it does earn its keep:
**throughput** when many users submit at once (each run gets its own container
regardless), and inputs well beyond the design notes' maximum. For one design
on the maximum input, use `design_single_container`.

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
