"""Serve the Streamlit portal from Modal, for an internal demo.

    modal deploy deploy/modal/portal.py

prints a URL. Share that with teammates; nothing else to set up.

WHY THIS AND NOT STREAMLIT COMMUNITY CLOUD / HUGGING FACE / RENDER
------------------------------------------------------------------
All of those need the model weights to reach them, and the weights are
deliberately out of git (public portal trajectory, provisional patent). Any of
them would mean either committing 2.8 MB of checkpoints to a repo a third party
clones, or standing up a secrets pipeline for a demo.

Modal already holds the weights, in the `lamp-design-weights` Volume, and
already runs the GPU function. So serving the UI from the same workspace adds
no new place for the weights to live, no new credential to manage, and no new
bill. The portal container here is a thin client: it submits to
`design_single_container` and polls, exactly as a Cloud Run deployment would.

ACCESS -- read this before sharing a link
-----------------------------------------
Two modes, and the choice is forced by what a browser can do.

**Default: Modal proxy auth.** ``requires_proxy_auth=True``; callers need a
proxy-auth token from this workspace. Strong, but the token goes in HTTP
headers (``Modal-Key`` / ``Modal-Secret``), and a browser cannot supply
headers for a link someone clicks. So this mode is right for scripts and for
your own use, and **not usable for teammates opening a link**.

**For a shareable demo: a public URL plus the app's password gate.**

    modal secret create lamp-design-portal-auth \
        LAMP_DESIGN_PORTAL_PASSWORD="$(openssl rand -base64 12)"

    LAMP_DESIGN_PORTAL_PUBLIC=1 modal deploy deploy/modal/portal.py

Then share the URL and the password. The gate lives in ``_gate_passed`` in
deploy/streamlit/app.py and is honest about what it is: one shared password in
front of a demo, not authentication. It identifies nobody and is exactly as
private as the people who know it.

Deploying public *without* the gate is refused at import time, because the
weights are kept out of git while the patent application is pending and a
public ungated URL would quietly undo that.

COLD START
----------
``min_containers=1`` keeps one instance warm, which is the same fix as Cloud
Run's ``min_instances = 1`` and the reason this does not take 20 seconds to
load. It bills a CPU container continuously -- cheap, since there is no GPU
attached here, but not nothing. Set it to 0 for a demo you use once, and
accept a slow first hit.

The GPU is only attached to the *design* function, which is a separate
container that exists for the seconds a run takes.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = os.environ.get("LAMP_DESIGN_PORTAL_APP", "lamp-design-portal")

#: Where the repo lands in the container.
REPO_REMOTE = "/app"

#: Streamlit's port inside the container. Modal proxies to it.
PORTAL_PORT = 8000

_THIS_FILE = Path(__file__).resolve()
LOCAL_REPO = (
    _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else Path(REPO_REMOTE)
)

#: Thin on purpose: no torch, no ViennaRNA, no MAFFT. None of them is executed
#: when compute is remote, and leaving them out is what keeps the image small
#: enough to start quickly. Same reasoning as
#: deploy/streamlit/perf/Dockerfile.thin, which documents it at length.
#:
#: `lamp_design` itself IS installed, because the UI uses it for input
#: validation, the geometry count and report rendering -- its model imports are
#: function-local, so it imports fine with no torch present.
portal_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "streamlit==1.65.0",
        "modal==1.6.3",
        "numpy==2.2.6",
        "pandas==2.2.3",
        "biopython==1.88",
        # Held below 2.1 like everywhere else: 2.1.0 shifts every Tm by
        # +2.17 C off the values the model was trained on. The UI only uses
        # primer3 indirectly, but the pin must not drift between images.
        "primer3-py==2.0.3",
    )
    .add_local_dir(str(LOCAL_REPO / "src"), f"{REPO_REMOTE}/src", copy=True)
    .add_local_dir(str(LOCAL_REPO / "deploy"), f"{REPO_REMOTE}/deploy", copy=True)
    .add_local_dir(str(LOCAL_REPO / "examples"), f"{REPO_REMOTE}/examples", copy=True)
    .add_local_file(str(LOCAL_REPO / "pyproject.toml"), f"{REPO_REMOTE}/pyproject.toml", copy=True)
    # --no-deps so pip cannot pull torch in behind our back, which would
    # quadruple the image for code that never runs here.
    .run_commands(f"cd {REPO_REMOTE} && pip install --no-deps -e .")
    .env(
        {
            # The UI submits to the GPU function instead of computing locally.
            "LAMP_DESIGN_BACKEND": "modal",
            "LAMP_DESIGN_MODAL_APP": os.environ.get(
                "LAMP_DESIGN_MODAL_APP", "lamp-design"
            ),
            "STREAMLIT_SERVER_HEADLESS": "true",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
            # Streamlit buffers an entire upload in memory before the app sees
            # a byte, so this cap protects memory, not just work.
            "STREAMLIT_SERVER_MAX_UPLOAD_SIZE": "5",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

#: Secret holding LAMP_DESIGN_PORTAL_PASSWORD, the shared password the app's
#: gate checks (see `_gate_passed` in deploy/streamlit/app.py).
#:
#: Modal has no concept of an optional Secret -- a missing one fails
#: `modal deploy` -- so it is attached only when asked for, the same pattern
#: deploy/modal/app.py uses for the weights secret. Create it once with:
#:
#:   modal secret create lamp-design-portal-auth \
#:       LAMP_DESIGN_PORTAL_PASSWORD="$(openssl rand -base64 12)"
PORTAL_SECRET_NAME = os.environ.get(
    "LAMP_DESIGN_PORTAL_SECRET", "lamp-design-portal-auth"
)

PORTAL_PUBLIC = os.environ.get("LAMP_DESIGN_PORTAL_PUBLIC") == "1"
USE_PASSWORD = PORTAL_PUBLIC or os.environ.get("LAMP_DESIGN_PORTAL_GATE") == "1"

# A public URL with no gate would put the portal -- and the designs it
# produces from weights kept out of git while the patent is pending -- on the
# open internet. Refuse at deploy time rather than discover it later.
if PORTAL_PUBLIC and os.environ.get("LAMP_DESIGN_PORTAL_NO_GATE") == "1":
    raise SystemExit(
        "refusing to deploy a public portal with the password gate disabled.\n"
        "Either drop LAMP_DESIGN_PORTAL_PUBLIC=1 (proxy auth, the default), or\n"
        "create the password secret and leave the gate on:\n"
        "  modal secret create lamp-design-portal-auth \\\n"
        '      LAMP_DESIGN_PORTAL_PASSWORD="$(openssl rand -base64 12)"'
    )

PORTAL_SECRETS: list = (
    [modal.Secret.from_name(PORTAL_SECRET_NAME)] if USE_PASSWORD else []
)

app = modal.App(APP_NAME, image=portal_image)


@app.function(
    # No GPU here. This container serves HTML and polls; the GPU lives in the
    # `lamp-design` app's design function and exists only during a run.
    cpu=1.0,
    memory=2048,
    # One warm instance, so a teammate opening the link does not wait for a
    # container to be built. This is the whole cold-start fix.
    min_containers=int(os.environ.get("LAMP_DESIGN_PORTAL_WARM", "1")),
    max_containers=int(os.environ.get("LAMP_DESIGN_PORTAL_MAX", "4")),
    # Streamlit holds a websocket open for the life of a session, so the
    # function must outlive a browsing session rather than a request.
    timeout=3600,
    scaledown_window=300,
    secrets=PORTAL_SECRETS,
)
@modal.concurrent(max_inputs=20)
@modal.web_server(
    PORTAL_PORT,
    # Streamlit needs a few seconds to boot before the first proxied request
    # succeeds; the 5s default is tight enough to produce a confusing 502 on a
    # cold container.
    startup_timeout=60.0,
    label=os.environ.get("LAMP_DESIGN_PORTAL_LABEL", "lamp-design-portal"),
    # Internal demo: callers need a Modal proxy-auth token from this
    # workspace. A leaked URL on its own gets nothing. See the module
    # docstring to make it public.
    requires_proxy_auth=not PORTAL_PUBLIC,
)
def portal() -> None:
    """Start Streamlit and let Modal proxy to it.

    ``web_server`` expects the process to be started and left running, not
    awaited -- Modal polls the port until it answers, then forwards traffic.
    """
    subprocess.Popen(
        [
            "streamlit",
            "run",
            f"{REPO_REMOTE}/deploy/streamlit/app.py",
            "--server.port",
            str(PORTAL_PORT),
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
            # Modal terminates TLS and proxies, so Streamlit must not try to
            # enforce its own CORS/XSRF origin checks against a hostname it
            # cannot see. Leaving these on produces a page that loads and then
            # fails to connect its websocket.
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "false",
        ],
        cwd=REPO_REMOTE,
    )
