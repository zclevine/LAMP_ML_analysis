# Streamlit web UI

Browser front end for the LAMP design engine: [`app.py`](app.py).

> **Not yet runnable end to end.** `streamlit` is an optional extra and is not
> installed in this repo's venv, so no page here has ever been rendered. Most
> engine stages it drives now exist; the two that do not
> (`thermo.compute_thermo`, `models.predict_tp`) fail with the name they
> wanted. The app's own logic — FASTA validation, run isolation, table and CSV
> building, rendering — is written and exercised against the real engine; see
> [What is verified](#what-is-and-is-not-verified).

## What it does

1. Takes a FASTA upload or pasted sequence.
2. Validates it **before any work starts**: size, FASTA shape, A/C/G/T-only
   alphabet, duplicate names, sequence count, minimum usable length, and the
   `n_targets × length` budget — using the engine's own limits, not a second
   set that can drift.
3. Exposes the parameters that are actually real: master mix (default **BST**,
   i.e. BstXT-dk), target concentration, number of sets to return (default 20),
   anchor stride and geometries per anchor. Tp threshold, minimum anchor
   separation, "already aligned" and the backend choice sit under *Advanced*.
4. Runs a design, locally in-process or on Modal.
5. Renders the ranked sets, the per-target Tp matrix, the mismatch matrix and a
   per-primer table, each downloadable as CSV. A local run also writes the
   engine's own `result.sets.csv` / `result.per_target.csv` / `result.json` /
   `result.summary.txt` into the run directory, identical to a CLI run's.
6. When nothing survives, shows the **funnel** from `DesignResult.funnel` and
   names the stage that emptied it — never a bare empty table.

Parameter defaults and bounds are read from the engine at import
(`report.DEFAULT_N_SETS`, `DEFAULT_TP_THRESHOLD_MINUTES`,
`DEFAULT_MIN_SEPARATION_BP`, `prepare.DEFAULT_MIN_LENGTH`, `MAX_SEQUENCES`,
`DEFAULT_MAX_BUDGET`, and `len(generation.DEFAULT_GEOMETRIES)`), so the widgets
cannot disagree with what the engine will accept.

## Run it

```bash
cd /home/ssiddiqu/lamp-design
.venv/bin/python -m pip install -e '.[streamlit]'

.venv/bin/streamlit run deploy/streamlit/app.py \
    --server.maxUploadSize 5 \
    --server.port 8080
```

`--server.maxUploadSize` (in MB) matters: **Streamlit buffers an entire upload
in memory before the app sees a single byte.** The app's own 5 MB check is what
stops oversized input from causing *work*; the server flag is what stops it
from causing *memory*. Set both, and keep them equal — the server default is
200 MB, which is far too generous for FASTA. (This is the one control that
cannot be enforced from inside `app.py`.)

### Environment

| Variable | Default | Meaning |
|---|---|---|
| `LAMP_DESIGN_DATA_DIR` | `<repo>/runs` | Root for persisted run outputs. **The seam for persistence** — point it at a mounted volume or GCS-backed path. |
| `LAMP_DESIGN_MAX_UPLOAD_MB` | `5` | Upload cap enforced in the app. Keep in step with `--server.maxUploadSize`. |
| `LAMP_DESIGN_MAX_BUDGET` | `prepare.DEFAULT_MAX_BUDGET` (50000) | `n_seqs × longest sequence` cap. |
| `LAMP_DESIGN_MAX_CANDIDATE_SETS` | `200000` | Passed to `generate_candidates(max_candidates=...)`. |
| `LAMP_DESIGN_WEIGHTS` | unset | Checkpoint directory for the local backend. Unset means `models.inference.resolve_weights_dir` resolves it. |
| `LAMP_DESIGN_ALIGN_THREADS` | `2` | MAFFT threads for a local run. |
| `LAMP_DESIGN_MODAL_APP` | `lamp-design` | Deployed Modal app name. Must match `APP_NAME` in `../modal/app.py`. |

## Backends

Chosen under *Advanced*:

* **`local`** — runs in the Streamlit process. Simplest, and the only option
  without a Modal account. It **blocks the session** for the whole run, so it is
  for a developer's laptop and small inputs, not for a shared deployment behind
  a request timeout. It runs `prepare` itself (so the UI can show panel QC
  before committing to a run) and then makes a single `lamp_design.design(...)`
  call. Earlier revisions re-implemented the stage sequence because no
  orchestration entrypoint existed; do not reintroduce that.
* **`modal-single`** — submits to `design_single_container`, then polls. One GPU
  container, thermodynamics on a multiprocessing pool.
* **`modal-fanout`** — submits to `design_fanout`. CPU fan-out for
  thermodynamics plus one GPU container; the scale-up path.

Both Modal backends are submit-and-poll: the app stores the call id, and a
**Check for results** button polls it. That is deliberate — a design takes tens
of seconds to minutes, so a blocking call would hold the session (and, on Cloud
Run, an HTTP request) open for the whole run. Polling is manual rather than an
auto-refresh loop because re-polling on every widget interaction buys nothing.

Deploy the Modal app first — see [`../modal/README.md`](../modal/README.md).

## Security and isolation

These are the points the team raised, and how each is handled.

**Upload size.** Capped in the app (`LAMP_DESIGN_MAX_UPLOAD_MB`) and at the
server (`--server.maxUploadSize`), for the two different reasons above. The
Modal functions re-check the cap server-side, because a client-side limit is a
convenience, not a control.

**Validate before working.** `validate_fasta()` is dependency-free and runs
before anything else. It rejects non-FASTA input, empty records and headers,
duplicate sequence names (they would collide in the Tp matrix, not merely look
odd), more than `prepare.MAX_SEQUENCES` (20) sequences, input where every
sequence is below `prepare.DEFAULT_MIN_LENGTH` (160 nt), anything over the
budget cap, and **any character other than A/C/G/T**. That last one matters:
the engine rejects ambiguity codes and gaps rather than cleaning them, because
RNAduplex and the model encoder accept nothing else — so a gate that quietly
accepted `N` would send the user all the way to a failed design. Gaps are
accepted only under "input is already aligned". Individual short sequences are
a warning, not a rejection: the engine excludes them from the panel and records
the reason, which the UI shows. It is a *gate*, not a replacement for
`prepare.py`'s QC, and it is checked against the engine's own
`utils.sequences.validate_sequence`.

**Per-run working directory.** Read
`/home/ssiddiqu/PrimerCast/gui/run_isolation.py` first: its docstring documents
the bug class — one Streamlit process serves many sessions, Cloud Run runs it
with concurrency > 1, and two runs sharing one working directory clobber each
other's control files and locks. Here:

* every run gets a private `tempfile.mkdtemp()` scratch directory, handed to the
  engine as an explicit `workdir=`, and removed in a `finally` whether the run
  succeeded or failed;
* outputs go to a separate persisted directory per run — the same split
  `prepare_run_dir` makes between private scratch and shared outputs;
* **the app never calls `os.chdir()`.** This is the important one. `os.chdir` is
  process-global, so in a multi-session Streamlit process it is strictly worse
  than the Snakemake case that motivated `run_isolation.py`: it would retarget
  the cwd of every *other* in-flight session, not just collide on one directory.
  There is no correct use of it in this file. An AST check for it is part of the
  scratch verification script (see below).

**One session's data must not be visible to another.**

* Uploaded sequences are written only inside that run's own directory — never
  into a shared input directory. PrimerCast puts uploads in a shared
  `target_seqs/original` that every session can then list; that is deliberately
  not copied here.
* Run ids are `<timestamp>-<uuid4 fragment>`, generated server-side and never
  user-supplied, so a run directory name is not guessable and there is no path
  component a user controls (hence nothing to traverse).
* The app renders a result only if its `run_id` is in this session's
  `st.session_state`. It never scans the runs directory, so there is no code
  path by which one session enumerates another's runs.
* Output directories are `chmod 0700`, since `mkdir(mode=...)` is masked by
  umask and a shared host may have other local users.
* No user-derived string is ever passed through `unsafe_allow_html`.

**Where results are persisted.** Under `LAMP_DESIGN_DATA_DIR/runs/<run_id>/`:
`envelope.json`, plus the engine's `result.sets.csv`, `result.per_target.csv`,
`result.json` and `result.summary.txt` for a local run. That is **local disk,
which is ephemeral on Cloud Run**: it is lost on redeploy and on scale events,
and a second instance cannot see the first's runs. The UI says so next to every
result. `LAMP_DESIGN_DATA_DIR` is the seam for pointing at durable storage
later — the same role `PRIMERCAST_DATA_DIR` plays in PrimerCast. With the Modal
backends there is also a durable copy in the `lamp-design-runs` volume.

## Deploying it

Not built here, and it should follow PrimerCast's pattern rather than invent a
new one (`/home/ssiddiqu/PrimerCast/CLAUDE.md`, "Web app deployment"): a
CPU-only image on Cloud Run, since Cloud Run has no GPU. The shape:

```dockerfile
CMD ["sh", "-c", "exec streamlit run \
    --server.port=${PORT:-8080} --server.address=0.0.0.0 \
    --server.maxUploadSize=5 \
    /app/deploy/streamlit/app.py"]
```

With that topology the UI is CPU-only and the GPU work belongs to Modal, so set
the backend default to `modal-single` there and keep `local` for development.

## What is and is not verified

Verified by a scratch script that stubs out `streamlit` and drives the module
against the **real** engine:

* the page body executing top to bottom with no input, raising nothing;
* FASTA parsing, and every validation rejection and warning above, including
  that `N` is rejected and that the engine's own `validate_sequence` agrees;
* gaps accepted only under `pre_aligned=True`;
* the sequence-count, budget, minimum-length and byte caps, against the
  engine's own constants;
* parameter defaults and bounds matching `report.DEFAULT_N_SETS` (20),
  `DEFAULT_TP_THRESHOLD_MINUTES` (30), `DEFAULT_MIN_SEPARATION_BP` (30),
  `prepare.DEFAULT_MIN_LENGTH` (160), `MAX_SEQUENCES` (20),
  `DEFAULT_MAX_BUDGET` (50000) and `len(generation.DEFAULT_GEOMETRIES)` (5);
* run ids unique over 300 draws and matching `<timestamp>-<8 hex>`;
* two runs getting distinct scratch and output directories, mode `0700`;
* no `chdir()` call anywhere in either deploy app (AST-checked);
* all four tables and their CSVs built from a **real**
  `report.result_to_dict` payload — the ranked-sets table carrying the eight
  primer columns in frozen `PRIMER_TYPE_ORDER`, the Tp and mismatch matrices
  keyed by `panel.target_names` with no holes, the per-primer long table with
  `+`/`-` strands;
* `render_envelope` running over that real payload without raising, and over an
  empty result, where it names `candidates_generated` as the stage that
  emptied the funnel.

Not verified:

* **Any actual rendering.** `streamlit` is not installed, so no widget, layout
  or download button has been exercised in a browser. The renderers were driven
  against a mock `st`, which proves the data plumbing, not the page.
* **A complete design.** `thermo.compute_thermo` and `models.predict_tp` do not
  exist, so `run_local` reaches them and fails with the name it wanted.
  Everything before and after them is exercised.
* **Both Modal backends**, which need a deployed app.
* **That `Condition.target_conc` is in copies per reaction.** The dropdown
  offers the values present in the training CSVs (10 … 1e8, default 10,000),
  and `encoding.encode_target_concentration`'s docstring confirms the scale
  ("100/1000/10000 -> -1/0/+1") — but the unit itself is an inference from the
  training data, not a documented contract.
* **Cloud Run behaviour**, including whether a local-backend run outlives the
  request timeout (it will not, for anything non-trivial — which is why the
  Modal backends exist).
