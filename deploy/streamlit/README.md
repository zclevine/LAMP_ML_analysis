# Streamlit web UI

Browser front end for the LAMP design engine. Two workflows, each a wizard with
a step indicator, built to navigate the same way as the in-house PrimerCast
portal (`/home/ssiddiqu/PrimerCast/gui/app.py`) so the two feel like one family:

* **Design** — Target → Parameters → Run → Results. Tiles candidate eight-primer
  sets across the whole input, scores every candidate against every input
  sequence, returns the top N positionally-diverse sets.
* **Evaluate** — Target → Primer Sets → Parameters → Run → Results. Scores
  primer sets you already have, and reports agreement metrics when the input
  carries a measured Tp.

## Run it

```bash
cd /home/ssiddiqu/lamp-design
.venv/bin/pip install streamlit

LAMP_DESIGN_WEIGHTS=/path/to/checkpoints \
.venv/bin/streamlit run deploy/streamlit/app.py \
    --server.headless true --server.port 8512 --server.maxUploadSize 5
```

`--server.maxUploadSize` (MB) matters: **Streamlit buffers an entire upload in
memory before the app sees a single byte.** The app's own 5 MB check is what
stops oversized input from causing *work*; the server flag is what stops it from
causing *memory*. Set both and keep them equal — the server default is 200 MB,
far too generous for FASTA. This is the one control that cannot be enforced from
inside the app.

## Layout

| File | Contents | Imports streamlit? |
|---|---|---|
| `app.py` | the five pages and the router | yes |
| `_layout.py` | step indicator, sidebar, navigation, preflight | yes |
| `_results.py` | laying result frames out, with downloads | yes |
| `_tables.py` | result payload → dataframes | no (pandas only) |
| `_backend.py` | the backend seam, run directories, the envelope | no |
| `_runner.py` | the out-of-process entry point `local` spawns | no |
| `_engine_api.py` | engine entry points and the bounds widgets read | no |
| `_validation.py` | input gates that run before any work | no |

The split is not cosmetic. Everything below `_layout` is importable without
Streamlit, which is what lets the runner subprocess and
`tests/test_streamlit_app.py` exercise the same code the pages run.

## Backends

One seam, two implementations, **the same envelope from both**, so no UI code
branches on where a run happened:

```
{schema: "lamp-design/modal-result@1", run_id, request, timing_seconds, result}
```

`result` is the engine's own `report.result_to_dict` for a design run, so there
is no second copy of the result schema to drift. An evaluate run carries
`lamp-design/evaluate-result@1` instead — per-row predictions and agreement
metrics, which is what `EvaluationResult` is; dressing that up as the design
payload would make the Results page lie about what it is showing.

* **`local`** (default, and the only one that works today) — spawns
  `_runner.py` on this host's GPU and polls its status file. Out of process, not
  in process, for three reasons that all bite in a long-lived server:
  `thermo.compute_dg` forks a multiprocessing pool and forking after an earlier
  run initialised CUDA in the same process is the classic hang; a crash in a
  native dependency cannot take the UI with it; and Stop can signal a process
  group but cannot interrupt a Python thread.
* **`modal-single` / `modal-fanout`** — `.spawn()` the deployed
  `design_single_container` / `design_fanout` and poll the call id. Deploy the
  Modal app first (`../modal/README.md`). **Unverified: there is no Modal
  account here.**

Selected with `LAMP_DESIGN_BACKEND=local|modal|modal-fanout` — an environment
variable, not a widget, because which compute a deployment may reach is an
operator's decision rather than a visitor's. The active backend is shown in the
sidebar. `modal` is not required for the app to run.

Both paths are submit-and-poll, and that is not a preference:
ARCHITECTURE.md measures the full sweep (5 kb × 10 targets × stride 1 × 4
conditions = 940,520 predictions) at **461 s on one L4**. No synchronous request
survives that, so the Run page submits, polls a status file every 2 s, and can be
left and returned to.

### Environment

| Variable | Default | Meaning |
|---|---|---|
| `LAMP_DESIGN_WEIGHTS` | unset | Checkpoint directory. **Unset is a preflight blocker**, not a late traceback. |
| `LAMP_DESIGN_BACKEND` | `local` | `local`, `modal` (= `modal-single`) or `modal-fanout`. |
| `LAMP_DESIGN_DATA_DIR` | repo root | Root for persisted runs (`<root>/runs/<run_id>`). **The seam for durable storage.** |
| `LAMP_DESIGN_MAX_UPLOAD_MB` | `5` | Upload cap enforced in the app. Keep in step with `--server.maxUploadSize`. |
| `LAMP_DESIGN_MAX_BUDGET` | `prepare.DEFAULT_MAX_BUDGET` (50000) | `n_sequences × longest` cap. |
| `LAMP_DESIGN_MAX_CANDIDATE_SETS` | `200000` | Passed to `generate_candidates(max_candidates=…)`. |
| `LAMP_DESIGN_MAX_PRIMER_SETS` | `200` | Cap on sets submitted to Evaluate. |
| `LAMP_DESIGN_ALIGN_THREADS` | `2` | MAFFT threads inside prepare. |
| `LAMP_DESIGN_MODAL_APP` | `lamp-design` | Must match `APP_NAME` in `../modal/app.py`. |

## What the pages do

**Target.** Upload or paste FASTA. `validate_fasta` runs before anything else
and rejects non-FASTA input, empty records and headers, duplicate names (they
would collide in the Tp matrix, not merely look odd), more than
`prepare.MAX_SEQUENCES`, input where every sequence is under
`prepare.DEFAULT_MIN_LENGTH`, anything over the budget cap, and **any character
other than A/C/G/T** — ambiguity codes are rejected rather than cleaned, because
RNAduplex and the model encoder accept nothing else. Gaps are accepted only under
"input is already aligned". One short sequence is a warning, since the engine
excludes it and records why. The page then says whether MAFFT will run: equal-
length ungapped rows are treated as an alignment already, which is the engine's
own rule.

**Primer Sets** (Evaluate only). A CSV/TSV with one column per primer
(`F3_sequence` or `F3`) goes to the engine **verbatim**, so the optional columns
`commands/evaluate.py` recognises keep working — `primer_set`, `target_seq`,
`master_mix`, `target_conc`, `<type>_target_sequence`, and `tp`, which turns the
run into a validation reporting Spearman, MAE and ROC-AUC. The page names the
optional columns it found. FASTA with `<set>_<type>` headers is also accepted and
converted to that CSV, because that is what someone with a handful of primers
has on hand.

**Parameters.** Grouped into expanders: reaction conditions (the engine's default
2 mixes × 2 concentrations, plus which one ranks), candidate generation (stride,
geometries), ranking and output (sets to return, minimum separation), scoring
(Tp threshold; for Evaluate also the window-mismatch limit and the censoring
value), and compute (device, thermo threads). Every default and bound is read
back out of the engine at import, so the widgets cannot disagree with what it
will accept. Underneath, an **estimated cost**: candidate sets, predictions, and
a wait derived from the measured 2,292 predictions/s on one L4 — so the Run page
is not where a user first learns a run takes eight minutes.

**Run.** Preflight first, and it blocks rather than warns: no input, no weights,
no aligner for ragged input, no ΔG backend, a ranking condition that is not being
scored. Cautions are separate (CPU-only host, stride 1, single target). Then a
stage checklist that fills in from the engine's own `progress` callback —
`generation 23,513 candidate sets`, `thermo 235,130 pairs`, `inference 940,520
predictions on cuda` — an elapsed clock, the live log, and the candidate funnel
on completion. Stop signals the runner's process group. The evaluate workflow's
stages are scraped out of the run log, because `commands/evaluate.py`'s `run`
owns its own progress callback; that is the same trick PrimerCast uses on
Snakemake's stdout.

**Results.** Headline metrics, the funnel, "what this run did" (candidate / pair
/ prediction counts, device, per-stage timings with each stage's share — all from
`DesignResult.params`), then tabs: ranked sets with the eight primer columns, Tp
by target, mismatches by target, every primer one per row, and the per-condition
sweep. Each is downloadable as CSV, alongside the engine's own
`result.sets.csv` / `result.per_target.csv` / `result.json` / `result.summary.txt`
exactly as a CLI run writes them. When nothing survives, the funnel is shown
first and **names the stage that emptied it** — never a bare empty table.

## Security and isolation

**Validate before working.** `_validation` is dependency-free and runs on every
rerun, before any engine call. It is a *gate*, not a replacement for
`prepare.py`'s QC, and it enforces the same rules so a user is never sent all the
way to a failed design by input the gate could have caught.

**Per-run directories, and no `os.chdir`.** Read
`/home/ssiddiqu/PrimerCast/gui/run_isolation.py` first: its docstring documents
the bug class — one process serving many sessions, two runs sharing a working
directory and clobbering each other's control files. Here every run gets a
private `tempfile.mkdtemp()` scratch directory, removed whether the run succeeded
or failed, and a separate persisted output directory. The app **never calls
`os.chdir()`**: it is process-global, so in a multi-session Streamlit process it
would retarget every *other* in-flight session's cwd. `tests/
test_streamlit_app.py` AST-checks that no module under `deploy/` calls it.

**One session's data is not visible to another.** Uploaded sequences are written
only inside that run's own directory, never a shared input directory (PrimerCast
puts uploads in a shared `target_seqs/original` that every session can list; that
is deliberately not copied here). Run ids are `<timestamp>-<uuid4 fragment>`,
generated server-side, so a directory name is not guessable and no path component
is user-controlled. Output directories are `chmod 0700`. The app renders a result
only if its run id is in this session's `st.session_state` — it never scans the
runs directory. No user-derived string reaches `unsafe_allow_html`.

**Where results live.** `<LAMP_DESIGN_DATA_DIR>/runs/<run_id>/`: `envelope.json`,
`status.json`, `run.log`, `run_config.json` and the engine's own output files.
That is local disk, **ephemeral on Cloud Run** — lost on redeploy and on scale
events, and invisible to a second instance. The UI says so next to every result.

## What is verified, and what is not

Verified on this host (one NVIDIA L4, `torch 2.14.1+cu126`, `streamlit 1.65.0`):

* **Real design runs end to end**, through `_backend.run_local` directly and
  through the full submit-and-poll path (`start_run` → `_runner.py` subprocess →
  `poll_run` → envelope), on GPU, with real predictions and all five output
  files written.
* **A real evaluate run** through `commands/evaluate.run`, including the
  measured-Tp validation metrics and the per-row predictions table.
* **Every page of both workflows rendering**, driven through Streamlit's own
  `AppTest` harness — the same `ScriptRunner` the server uses — including the
  polling loop driving reruns until a run completes, and the results tabs.
* The server returning **HTTP 200** with the page HTML, and `/_stcore/health`
  returning `ok`.
* 49 tests in `tests/test_streamlit_app.py`.

Not verified:

* **Both Modal backends.** There is no Modal account here: nothing has been
  deployed, submitted or polled. The one place to widen if polling misreports a
  still-running call is the `except TimeoutError` in `_backend._poll_modal`.
* **A multi-condition remote run.** `deploy/modal/app.py`'s `_validate_request`
  accepts a single master mix and concentration, so a remote run scores only the
  ranking condition. The Parameters page says so rather than letting the result
  quietly contain fewer conditions than were asked for.
* **A browser.** No human has clicked through this; `AppTest` exercises the
  script and the widget tree, not the rendered pixels.
* **MAFFT.** It is not installed on this host, so no run here has been aligned.
  Ragged multi-sequence input is a preflight blocker that says exactly that.
* **Cloud Run.** Nothing is deployed. With no GPU there, the backend default
  would want to be `modal-single`.
* **That `target_conc` is in copies per reaction.** The dropdown offers the
  values present in the training CSVs, and
  `encoding.encode_target_concentration`'s docstring confirms the scale
  ("100/1000/10000 → -1/0/+1") — but the unit itself is an inference from the
  training data, not a documented contract.

## Deploying it

Not built here, and it should follow PrimerCast's pattern rather than invent one
(`/home/ssiddiqu/PrimerCast/CLAUDE.md`, "Web app deployment"): a CPU-only image
on Cloud Run, since Cloud Run has no GPU, with the GPU work on Modal.

```dockerfile
CMD ["sh", "-c", "exec streamlit run \
    --server.port=${PORT:-8080} --server.address=0.0.0.0 \
    --server.maxUploadSize=5 \
    /app/deploy/streamlit/app.py"]
```

With that topology set `LAMP_DESIGN_BACKEND=modal-single` there and keep `local`
for development.
