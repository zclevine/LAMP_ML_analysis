# Notes to Bryan

Drafts in Sameed's voice — edit freely before sending.

The framing changed on 2026-11-02: the model weights now ship in the repo
(2.8 MB), so the handoff is "clone it" and Docker is the lead option rather
than the fallback. Modal is still there and still faster, but it is no longer
something Bryan has to learn in order to run this.

---

## (A) The main one — here's the repo, it should just run

> Hey Bryan — the LAMP design tool is ready for you to look at, and I think
> I've managed to make it boring to set up, which was the goal.
>
> Everything is in the repo, including the model. The five-seed ensemble is
> only 2.8 MB so I just committed it — there's nothing to download, no mount,
> no environment variable, and nothing you need to ask me for. `docker build`
> and it works, or `pip install -e .` and `lamp-design design --in
> examples/H5N1_HA_demo.fasta` if you want to see it run without any of the
> web stuff. There's a demo FASTA in the repo so you don't have to go find
> sequences.
>
> The image is built the way PrimerCast's is (micromamba, deps in one solve,
> tools verified at build time) and the portal image matches your Cloud Run
> contract exactly — listens on `$PORT`, binds 0.0.0.0, headless — so it
> should drop into your Terraform without you inventing anything. Honestly
> yours is the model we've been copying throughout: same Streamlit wizard,
> same submit-and-poll shape.
>
> On speed, so you can size it: a full run scoring every position of a ~1.7 kb
> target against ten genomes is about 4 minutes on 8 CPU cores, which
> extrapolates to roughly 8 minutes on your 4 — inside your 600 s timeout, but
> only by about 20%, and that's an extrapolation rather than a measurement. It
> peaks at 3.4 GB against your 4 GiB. So it fits, but I'd want you to measure
> it with a real panel before we build on it. `--stride` is the lever if it's
> tight; it trades candidate coverage for time roughly linearly.
>
> There's a doc in there written for whoever does the integration —
> `docs/for_bryan/INTEGRATION.md` — with the contracts, the measured numbers,
> and an honest list of the things that will bite. Point your agent at it if
> that's easier.

---

## (B) If you want it fast — the GPU options

> Two ways to make a run take seconds instead of minutes, and I genuinely
> don't mind which you pick.
>
> **Cloud Run GPU.** I only recently clocked that this exists — L4s, generally
> available, no quota request for the first few, and it scales to zero. That
> means you can stay entirely inside Google with the same container: `--gpu 1
> --gpu-type nvidia-l4` and you're done. We measured an L4 at about 37 seconds
> for a full run. Two things to know: GPU forces instance-based billing, so
> you pay for the instance's life rather than per request, and it needs 4 CPU +
> 16 GiB alongside. Also turn off zonal redundancy, it's on by default and
> costs more.
>
> **Our Modal setup.** We have the GPU side deployed on Modal already, which
> is a bit faster (L40S, ~19 s) and bills per second with no instance
> overhead. If you'd rather not add a third-party dependency, ignore this
> entirely — but if you want it, your container needs only Streamlit and
> `requests` and talks to two HTTP endpoints. I wouldn't hand you a Modal
> token, since they're workspace-scoped and effectively a key to our whole
> account, so there's a rotatable API key instead. Say the word and I'll make
> you one.
>
> My read: if Broad would rather own the whole thing, Cloud Run GPU is the
> cleaner answer and I'd support that. The Modal path exists because it let us
> get numbers quickly, not because the design depends on it — the compute
> backend is a swappable seam, so this isn't a decision that locks anything in.

---

## (C) The slow first load

> One more thing, unrelated to our stuff — PrimerCast takes about 21 seconds
> to load for me on a fresh browser and about 3 on a reload, and I had my
> agent dig into why since we'd inherit the same problem. It's not your app
> code.
>
> It's that Cloud Run is scaled to zero, so the first visit after any idle
> period has to spin up the whole container — mount the image, start Python,
> import torch, boot Streamlit — before it sends a single byte. And because
> Cloud Run holds the request open while that happens, the browser has nothing
> at all to render: not a spinner, not a logo. Just a blank white tab that
> looks exactly like a dead site. I'd assumed the app was broken more than
> once before I understood this.
>
> Cheapest fix is a Cloud Scheduler job pinging `/_stcore/health` every five
> minutes, which keeps an instance alive for almost nothing; `min_instances =
> 1` does it deterministically but you pay for an always-on container. One
> wrinkle worth checking: the cheap version only avoids the always-on bill if
> `cpu_idle = true`, and I think your config has it false — with CPU always
> allocated you pay for the instance's whole life either way. Whether
> Streamlit is happy being throttled between sessions I'd want tested rather
> than assumed.
>
> We also found the frontend bundle ships uncompressed (3 MB where gzip gives
> 730 KB, and Google Frontend passes it through) — real, but the smaller
> effect, worth under a second on a decent connection. Streamlit bypasses gzip
> on purpose and expects a proxy or CDN to do it, so it's a deployment gap
> rather than a bug. There's a working nginx config in our repo
> (`deploy/streamlit/perf/`) if it's ever worth it, plus a stripped-down image
> that drops torch entirely for the thin-client case — 883 MB instead of
> 3.4 GB, which attacks the cold start at the source instead of paying to
> paper over it.

---

## (D) Internal demo link, for teammates

> If you want to poke at it before any of this is decided — it's deployed and
> I can send you a link plus a password. Takes nothing to stand up on our side,
> so it's the easiest way to look at the actual thing rather than screenshots.
