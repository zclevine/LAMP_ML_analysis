# Running LAMP-ADAPT

Everything needed is in this repository, including the trained model (2.8 MB).
There is nothing to download, no account to create, and no configuration.

Two ways to run it. Pick one.

---

## A. The web portal (what most people want)

```bash
git clone <this repo>
cd lamp-design

docker build -t lamp-design:cpu .
docker build -t lamp-design:portal -f Dockerfile.portal .

docker run --rm -p 8080:8080 lamp-design:portal
```

Open <http://localhost:8080>. On the Target step choose **Demo panel** to try it
without supplying sequences of your own.

The first build takes several minutes (it solves a conda environment and
verifies the thermodynamics tools). Subsequent builds are cached.

### If you do not use Docker

```bash
python -m pip install -e ".[streamlit]"
streamlit run deploy/streamlit/app.py
```

You will also need ViennaRNA's `RNAduplex` on your `PATH` — `conda install -c
bioconda viennarna`, or `apt install vienna-rna` on Debian/Ubuntu. The Docker
route exists precisely to avoid this step.

---

## B. The command line (for batch work or a pipeline)

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data \
  lamp-design:cpu \
  lamp-design design --in /app/examples/H5N1_HA_demo.fasta --out /data/demo
```

`--user` matters: the image runs as a non-root user, so without it every write
to `/data` fails with "Permission denied" *after* the run has finished its work.

Four files are written: `demo.sets.csv`, `demo.per_target.csv`, `demo.json`
and `demo.summary.txt`.

---

## What the input has to look like

**A FASTA of pre-aligned, equal-length sequences with no gaps**, using only
A, C, G and T.

- Every sequence the same length.
- No `-` characters. A gapped alignment is refused, not repaired: a primer
  cannot sit over a column that most of the panel does not cover.
- No ambiguity codes (`N`, `R`, `Y`, …). These are rejected rather than
  silently substituted, because substituting would change the design without
  telling you.
- Ragged input is refused rather than aligned. Align and trim it yourself
  first — aligning here would introduce the gap columns above.

`examples/H5N1_HA_demo.fasta` is a worked example of a conforming panel: ten
2025 H5N1 hemagglutinin genomes, MAFFT-aligned and then trimmed to the largest
contiguous gapless block (1,715 of 1,751 columns).

If your sequences are ragged, the usual recipe is: align them, then cut to a
region where every sequence has a base.

---

## What it does, and what it gives you

Every position along the alignment is tried. At each one, five candidate
eight-primer LAMP sets are laid down — the canonical geometry (spacings
5/0/0/5/0/0/5/0 at the median primer lengths of the training corpus, where the
model is best calibrated) plus four whose lengths and spacings are drawn from a
normal distribution over the permitted ranges. On a 5 kb target that is roughly
25,000 candidates.

Each candidate gets real thermodynamics (primer3 Tm, ViennaRNA binding ΔG) and
is then scored against **every** sequence you supplied by a five-seed neural
ensemble. The twenty best are returned, spread across the target so the list is
not one design repeated twenty times.

**The output is a ranking.** Rank 1 is the best candidate the model found. A
banded speed estimate is included for comparison between runs; anything in the
fastest band reads `<3 min`, because the model does not resolve differences
inside it.

---

## How long it takes

| | |
|---|---|
| 8 CPU cores | ~4 minutes, measured |
| 4 CPU cores | ~8 minutes, extrapolated — measure it yourself |
| 1 GPU | ~40 seconds |

Peak memory is 3.4 GB. If you are deploying behind a request timeout, measure
with a real panel before relying on the extrapolated figure, and use `--stride`
to trade coverage for time (it scales roughly linearly).

GPU is optional and off by default. See [`docs/HANDOFF.md`](docs/HANDOFF.md) if
you want it.

---

## Checking the install

```bash
pytest -q
```

522 tests, no skips. The suite includes the trained model, so a green run means
the prediction path was actually exercised rather than skipped.

To check only that the tools agree with the version the model was trained
against:

```bash
python scripts/check_dg_reference.py
```

This compares six fixed primer pairs against binding energies measured on
ViennaRNA 2.4.17 and exits non-zero on drift. A different ViennaRNA does not
crash — it quietly returns different numbers — which is why this check exists.

---

## Known limitations

1. **No off-target or specificity screening.** The tool optimises for
   amplification speed and panel coverage. It does not check whether a primer
   set also amplifies something you did not ask about.
2. **No dimer or hairpin filter**, by decision rather than oversight — the
   training corpus was not screened for it either.
3. **Design quality is unvalidated at the bench.** Rankings are reproducible
   and the model reproduces its published accuracy on pathogens absent from
   training, but "the top-ranked set amplifies well" needs wet-lab
   confirmation.

---

## Getting help

- [`docs/HANDOFF.md`](docs/HANDOFF.md) — deployment, including GPU and Cloud Run
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — why the engine is built this way
- [`docs/for_bryan/INTEGRATION.md`](docs/for_bryan/INTEGRATION.md) — written for
  a coding agent wiring this into an existing deployment
