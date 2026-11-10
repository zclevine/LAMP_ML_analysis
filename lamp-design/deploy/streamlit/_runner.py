"""Out-of-process entry point for the ``local`` backend.

``python _runner.py <request.json> <work_dir>``

Started by ``_backend.start_run`` and never imported by the app. It owns one
run: it calls the engine, writes the stage-by-stage status the Run page polls,
leaves the envelope in the run directory, and removes its own scratch
directory whether it succeeded or failed.

Why a subprocess rather than a thread in the Streamlit process:
``thermo.compute_dg`` forks a multiprocessing pool, and forking a process that
has already initialised CUDA (which an earlier run in the same process would
have done) is the classic hang; a native crash here cannot take the UI with it;
and Stop can signal a process group but cannot interrupt a Python thread.

Everything it prints is the run log. The engine's own stage lines arrive here
through the ``progress`` callback, so the log and the checklist cannot disagree.
"""

from __future__ import annotations

import json
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _backend import (  # noqa: E402  (path set up above, deliberately)
    STATUS_SCHEMA,
    RunCancelled,
    RunPaths,
    envelope_path,
    evaluate_local,
    out_dir_for,
    public_request,
    read_status,
    run_local,
    save_run_config,
    write_status,
)


def _install_stop_handler() -> None:
    """Turn SIGTERM into an exception so the ``finally`` blocks still run.

    ``stop_run`` signals this process group. Dying on the default handler would
    leave ``status.json`` saying "running" for ever and leak the scratch
    directory, so the signal is converted into a normal Python error instead.
    """

    def _raise(signum: int, frame: Any) -> None:
        raise RunCancelled("stopped by the user")

    signal.signal(signal.SIGTERM, _raise)
    signal.signal(signal.SIGINT, _raise)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {Path(argv[0]).name} <request.json> <work_dir>", file=sys.stderr)
        return 2

    _install_stop_handler()
    request = json.loads(Path(argv[1]).read_text())
    work_dir = Path(argv[2])
    run_id = request["run_id"]
    workflow = request.get("workflow", "design")
    paths = RunPaths(run_id=run_id, out_dir=out_dir_for(run_id), work_dir=work_dir)

    started = time.time()
    status: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "run_id": run_id,
        "workflow": workflow,
        "backend": "local",
        "state": "running",
        "pid": None,
        "started_at": started,
        "updated_at": started,
        "stages": [],
        "request": public_request(request),
    }
    import os

    status["pid"] = os.getpid()
    write_status(run_id, status)
    print(f"run {run_id} ({workflow}) starting", flush=True)

    def progress(stage: str, message: str) -> None:
        elapsed = time.time() - started
        status["stages"].append(
            {"stage": stage, "message": message, "elapsed_s": round(elapsed, 2)}
        )
        status["stage"] = stage
        status["updated_at"] = time.time()
        write_status(run_id, status)
        print(f"[{elapsed:7.1f}s] {stage:<10} {message}", flush=True)

    try:
        runner = run_local if workflow == "design" else evaluate_local
        envelope = runner(request, paths, progress)
    except RunCancelled as exc:
        status.update(state="cancelled", error=str(exc), finished_at=time.time())
        write_status(run_id, status)
        save_run_config(request, backend="local", status="cancelled")
        print(f"cancelled: {exc}", flush=True)
        return 130
    except BaseException as exc:  # a failed run must report, never vanish
        status.update(
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=time.time(),
        )
        write_status(run_id, status)
        save_run_config(request, backend="local", status=f"failed ({type(exc).__name__})")
        traceback.print_exc()
        return 1
    finally:
        # Always remove the scratch directory. Only the persisted out_dir
        # survives, and it never held the user's sequences in the first place.
        shutil.rmtree(work_dir, ignore_errors=True)

    envelope_path(run_id).write_text(json.dumps(envelope, indent=2))
    result = envelope["result"]
    # A design run returns ranked sets; an evaluate run returns scored rows.
    outcome = (
        f"{result.get('n_scored', 0):,} row(s) scored"
        if result.get("kind") == "evaluate"
        else f"{result.get('n_sets_returned', 0)} set(s) returned"
    )
    status.update(
        state="succeeded",
        finished_at=time.time(),
        timing_seconds=envelope.get("timing_seconds", {}),
        outcome=outcome,
    )
    write_status(run_id, status)
    save_run_config(request, backend="local", status="succeeded")
    print(f"done in {time.time() - started:.1f}s: {outcome}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
