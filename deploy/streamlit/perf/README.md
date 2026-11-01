# Why the portal is slow to load, and what fixes it

> Nothing in this directory is used by the default deployment. `Dockerfile.portal`
> at the repo root is the plain image the main docs describe. This is an
> optional variant plus the measurements that justify it, so the decision can
> be made rather than inherited.

The complaint is real and not specific to this app. The observed figures, from
the production service:

| | |
|---|---|
| fresh browser, cold cache | **~21 s** |
| same browser again | **~3 s** |

The ~3 s is the floor with every asset already cached: websocket handshake,
one script run, render. So **~18 s is what a first visit pays on top**, and the
job of this document is to say what that 18 s is actually made of.

### Ranked by what it is likely to be

Measured from a GCP VM in the same region as the service, so network is close
to ideal and what remains is structural:

| suspect | measured | likely share of the 18 s |
|---|---|---|
| **cold start** (`min_instances = 0`) | not directly measurable from outside | **probably most of it** |
| **68 separate requests** | 4.50 s of summed per-request latency | seconds, worse on a high-latency link |
| **3 MB of JS to parse** | client CPU, not network | seconds on a modest laptop |
| **uncompressed bytes** | 3,098 KB vs 732 KB gzipped | 0.8 s at 25 Mbps, 3.9 s at 5 Mbps |

**Compression is the smallest of the four on a decent connection**, which is
the opposite of where I started. It is still worth doing — it is free once a
proxy or CDN is in place — but `min_instances = 1` is the change to make first,
and it is one line.

### The 30-second test that settles it

Cold start and asset cost are indistinguishable from the outside, but not from
a browser:

1. Load the site once so the Cloud Run instance is warm.
2. Immediately open DevTools, tick **Disable cache**, and hard-reload.

That is a first-visit asset load against a *warm* instance.

- Still ~18-21 s → it is assets and JS parse. Compression and the proxy help.
- Drops to ~5 s → the 21 s was mostly **cold start**. Set `min_instances = 1`
  and the rest is a rounding error.

---

## Cause 1: the frontend ships uncompressed (76% of bytes, but see the ranking above)

Measured on Streamlit 1.65.0, which is the build PrimerCast also runs — same
asset hash, `index.CKCTizkM.js`:

| | |
|---|---|
| first-load payload, 68 files | **3,098 KB** |
| the same content, gzip -9 | **732 KB** |
| wasted on every first visit | **2,366 KB (76%)** |

Largest single offenders: `protobuf` 813 KB, `pandasStylerUtils` 735 KB,
`index` 439 KB.

Confirmed on the live deployment — gzip offered, gzip refused:

```console
$ curl -s -H "Accept-Encoding: gzip, br" -o /dev/null -D - \
    https://primercast.sabeti.broadinstitute.org/static/js/index.CKCTizkM.js
content-length: 449225          # no content-encoding header
```

Google Frontend in front of Cloud Run passes those bytes through unchanged.
Cache headers are already correct (`cache-control: public, immutable,
max-age=31536000`), so this is purely a **first** visit cost — the visit that
decides whether someone keeps using the tool.

### This is Streamlit's deliberate choice, not a bug

`streamlit/web/server/starlette/starlette_gzip_middleware.py` bypasses gzip on
`/static/` on purpose, and the reasoning is sound:

- Starlette's `GZipMiddleware` re-compresses the body on **every request** and
  never caches the output, and uvicorn has no zero-copy sendfile path — so the
  origin burns CPU per hit for a bandwidth saving each client collects once.
- Under concurrency their load testing found the bypass *improved* initial load
  time and peak RSS.
- And then the sentence that matters here: *"Production usually sits behind a
  reverse proxy / CDN that compresses and caches these responses."*

**That proxy is the missing piece.** Forcing compression inside the Python
process is fighting upstream for a reason they already evaluated and rejected —
I tried it first and threw it away. The fix belongs in front.

### Fix A — Cloud CDN in front of Cloud Run (recommended for Broad infra)

Compression and caching at the edge, zero cost in the container, nothing in
this repo to maintain. If Cloud CDN is available on the project, this is
strictly better than Fix B.

### Fix B — a compressing, caching proxy inside the container

[`nginx.conf`](nginx.conf) + [`Dockerfile.proxied`](Dockerfile.proxied) +
[`start.sh`](start.sh), for when a CDN is not on the table.

```bash
docker build -t lamp-design:cpu .
docker build -t lamp-design:portal -f Dockerfile.portal .
docker build -t lamp-design:portal-fast -f deploy/streamlit/perf/Dockerfile.proxied .
docker run --rm -p 8080:8080 -v /path/to/weights:/weights:ro lamp-design:portal-fast
```

**Verified by running it**, 2026-11-01:

| | |
|---|---|
| `index.js`, gzip offered | **137,134 B** (was 449,225) |
| `index.js`, gzip refused | 449,225 B — still correct |
| response headers | `Content-Encoding: gzip`, `X-Cache-Status: HIT` |
| full first load | **3,098 KB → 741 KB (76%)** |
| `/_stcore/stream` upgrade | **HTTP 101** |
| healthy after | 2 s |

`proxy_cache` is the part that answers Streamlit's objection: `X-Cache-Status:
HIT` means each asset is compressed **once** and then served from cache, so
the CPU cost does not recur per request. Three details that are easy to get
wrong and are handled:

- `/_stcore/stream` is proxied with `proxy_buffering off` and a 3600s timeout.
  That websocket carries every interaction; at proxy defaults the app connects
  and then goes blank.
- `woff2` is excluded from `gzip_types` — already compressed, so re-gzipping
  burns CPU and can grow it.
- `start.sh` uses `wait -n`, so the container exits if *either* process dies.
  A proxy whose backend has gone is worse than no container at all, because
  health checks keep passing while every request 502s. It also has a
  `#!/bin/bash` shebang on purpose: `wait -n` is a bash builtin and the base
  image's `/bin/sh` is dash, which exits 2 with "Illegal option -n" before
  serving anything. Found by running the image, not by reading it.

---

## Cause 2: cold starts (probably the bigger one)

PrimerCast sets `cloud_run_min_instances = 0`, with the comment *"0 = scale to
zero (accept cold starts for this low-frequency app)"*. For a low-traffic tool
that means **most visits are the first visit**: the container is pulled,
Python starts, and the ML stack imports before a single byte is served.

For scale, our portal image is 3.4 GB and takes ~2s to serve its first request
once the image is already local — on Cloud Run, add the image pull and a cold
Python start on top of that.

### What a cold start actually is

Not a session or a connection — **the whole server does not exist yet.** Cloud
Run runs nothing when idle. The first request after idle has to, in order:

1. allocate a machine and mount the container image (GBs, and the ML stack is
   most of it),
2. start a fresh Python interpreter,
3. `import torch` — **1.76s measured**, before any application code,
4. import Streamlit and the app's own modules — ~1.2s measured,
5. boot the Streamlit server and run the script once,
6. *then* send the first byte of HTML.

Only after all six does the browser even learn which assets to fetch. So this
is not "the page is slow to load" as distinct from the server — the page is
slow to load *because* the server is still being built while the browser
waits. Every later visit within the idle window skips all six.

### Fix: `min_instances = 1`

One always-warm instance, one line of Terraform. Highest-impact change
available. The cost is a container billed continuously rather than per
request, which is exactly the tradeoff PrimerCast's `min_instances = 0`
comment was making — reasonable for a tool nobody is being pointed at, wrong
for a demo people are being *invited* to try. Check current Cloud Run pricing
for the configured 4 vCPU / 4 GiB before committing; it is not free.

### Cheaper alternative: keep it warm with a ping

A Cloud Scheduler job hitting `/_stcore/health` every 5 minutes keeps the
instance alive at the cost of a trivial request, rather than paying for a
reserved instance. Less deterministic than `min_instances` — a burst can
still cold-start a *second* instance — but it removes the common case for
roughly nothing, and `/_stcore/health` is already there for it.

`startup_cpu_boost = true` and `cpu_idle = false` are already set in
PrimerCast's `cloud_run.tf`, so that lever is pulled.

---

## What is *not* the cause

Measured, so these can be ruled out rather than argued about:

| suspected | measured | verdict |
|---|---|---|
| server slow to respond | initial HTML **6.4 KB in 3 ms** | not it |
| heavy Python imports per page | `_backend` 0.29s, `_layout` 0.87s | not it |
| torch imported at page load | **not in `sys.modules`** after importing the app | not it |
| our app code | PrimerCast has the identical problem | not it |

The app deliberately keeps torch out of module scope, so a page view never
pays a 1.76s torch import. That part is already right and worth not breaking.

---

## Summary for a deployment decision

| change | effort | effect |
|---|---|---|
| **`min_instances = 1`** | one Terraform line | removes cold start — **do this first** |
| Cloud CDN in front | infra config | −76% bytes, no container cost |
| `Dockerfile.proxied` | use this image | −76% bytes, ~30 MB image, 2 processes |

**Do `min_instances = 1` regardless**, then run the browser test above before
spending effort on the other two — it tells you whether they are worth it for
this audience's connections. Expect it to matter a lot for someone on hotel
wifi and barely at all on campus.

Both compression options leave the 68-request count and the 3 MB JS parse
untouched. Those are Streamlit's frontend, not something this repo can fix;
the only lever on them is not using Streamlit, which is not worth it here.
