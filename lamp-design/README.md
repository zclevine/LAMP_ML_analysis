# LAMP-ADAPT

ML-guided design of eight-primer LAMP primer sets.

Give it a set of aligned target sequences. It tiles every position with
candidate primer sets, computes real thermodynamics for each, scores every
candidate against every one of your sequences with a trained neural ensemble,
and returns the twenty best — ranked, and spread across the target so the list
is not one design repeated.

The trained model ships in this repository (2.8 MB), so a clean clone runs with
nothing to download and nothing to configure.

> The package, CLI and Docker images are named `lamp-design`; the tool is
> LAMP-ADAPT in the manuscript. Same thing.

---

## Quickstart

```bash
docker build -t lamp-design:cpu .
docker build -t lamp-design:portal -f Dockerfile.portal .
docker run --rm -p 8080:8080 lamp-design:portal
```

Open <http://localhost:8080> and pick **Demo panel** on the Target step to try
it without supplying your own sequences.

Or from the command line, with no server:

```bash
pip install -e .
lamp-design design --in examples/H5N1_HA_demo.fasta --out /tmp/demo
```

Full instructions, including the non-Docker route, are in
**[SETUP.md](SETUP.md)**.

---

## Input

**A FASTA of pre-aligned, equal-length, gapless sequences, A/C/G/T only.**

Gapped alignments, ragged input and ambiguity codes are refused rather than
repaired — a primer cannot sit over a column most of the panel does not cover,
and silently substituting an `N` would change the design without saying so.
`examples/H5N1_HA_demo.fasta` is a worked example of a conforming panel.

## Output

A **ranking**. Rank 1 is the best candidate found. Each set comes with its
primer sequences, position, per-target mismatch counts, panel coverage, and a
banded speed estimate (the fastest band reads `<3 min`, because the model does
not resolve differences inside it).

## Speed

| | |
|---|---|
| 8 CPU cores | ~4 min, measured |
| 4 CPU cores | ~8 min, extrapolated |
| 1 GPU | ~40 s |

CPU-only and self-contained by default. GPU is optional.

---

## Documentation

| | |
|---|---|
| **[SETUP.md](SETUP.md)** | Running it, from a clean clone |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deploying it: contracts, Cloud Run, GPU, measured numbers |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Why the engine is built this way |

## A note on CPU

**Fusion is not recommended on CPU.** A fusion run takes roughly five hours,
and that figure does not shrink with a smaller input: TabPFN re-encodes its
34,283-row training context on every call, so three primer sets cost about
the same as three hundred. On a GPU the same run is ~90 seconds.

Simplified Lyra, the default, is fine on CPU: a full design run is ~41 min
on 8 cores, and an evaluate run is under a second.

## Credits

The optional Fusion model includes **TabPFN-2** as one of four branches.
Built with PriorLabs-TabPFN. Those weights are distributed under the Prior
Labs License (Apache 2.0 with an additional attribution provision), a copy of
which is in [`third_party/LICENSE-PriorLabs-TabPFN`](third_party/LICENSE-PriorLabs-TabPFN).

## Limitations

1. **No off-target or specificity screening.** The tool optimises amplification
   speed and panel coverage; it does not check what else a set might amplify.
2. **No dimer or hairpin filter**, by decision — the training corpus was not
   screened for it either.
3. **Design quality is unvalidated at the bench.** Rankings are reproducible and
   the model reproduces its published accuracy on pathogens absent from
   training, but "the top-ranked set amplifies well" needs wet-lab confirmation.

## Tests

```bash
pytest -q
```

522 tests, no skips. The trained model is included, so a green run means the
prediction path was exercised rather than skipped.
