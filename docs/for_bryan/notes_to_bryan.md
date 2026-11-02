# Notes to Bryan

Drafts in Sameed's voice — edit freely before sending.

---

## (A) The Modal approach

> Hey Bryan — quick update on where the LAMP design tool has landed, and a
> question for you at the end.
>
> The short version is that the compute side is done and running. We ended up
> putting the GPU work on Modal rather than trying to fit it into a container,
> because the numbers pushed us there pretty hard: a full design run — every
> position in a 5 kb target, five primer-set layouts per position, scored
> against every sequence in the panel — is about 84 GPU-seconds on one L40S,
> which works out to roughly 5 cents a run. The same job on 8 CPU cores takes
> 4 minutes. It's deployed in our Modal workspace now and we've checked that
> the remote results match a local run to six decimal places, so it's not an
> approximation of the thing, it *is* the thing.
>
> What I'd love is for the portal to look and work like PrimerCast, because
> honestly yours is the model we've been copying — same Streamlit wizard, same
> submit-and-poll shape, and we've matched your Cloud Run contract exactly
> (listens on `$PORT`, binds 0.0.0.0, headless) so it should drop into your
> Terraform without you inventing anything new. The one difference is that
> instead of the container doing the maths, it submits to Modal and polls.
>
> The bit I want your opinion on: I don't want to hand you a Modal token,
> because they're workspace-scoped and that's effectively a key to our whole
> account. So we built an HTTP front door with a rotatable API key instead —
> your container would need only Streamlit and `requests`, no torch, no model
> weights, nothing of ours to maintain. Does that work on your side, or would
> you rather run `modal deploy` in your own workspace and own the bill? Both
> are supported and I genuinely don't mind which. I've written all of it up
> for whoever (or whatever) does the integration.

---

## (B) The Docker approach

> On Docker — we do have a proper CPU-only image as well, and I want to be
> straight with you about where it does and doesn't fit, because I think the
> answer changed last week.
>
> The image is built the same way as PrimerCast's (micromamba, deps in one
> solve, tools verified at build time) and it runs the whole pipeline with no
> external dependencies at all — useful for Snakemake and batch, and it's the
> escape hatch if Modal ever becomes a problem. What held it back was speed: a
> full stride-1 run was ~15 minutes on 8 cores, which doesn't fit your Cloud
> Run request timeout, and it peaked at 4.1 GB against your 4 GiB limit, so it
> was failing on both axes at once.
>
> Then we found that a lot of the work was redundant. When you score one
> primer set against ten closely-related genomes, most of those genomes are
> *identical* across the ~170 bases the primers actually touch — so the model
> was being handed byte-identical input and recomputing the same answer. We now
> collapse those, which is exact rather than an approximation, and it cut the
> CPU run from 15 minutes to 4 and the memory to 3.4 GB. On your 4 cores that
> extrapolates to roughly 8 minutes, which is *inside* your 600-second timeout
> — but only by about 20%, and that's an extrapolation rather than a
> measurement, so I wouldn't want to build on it without you testing with a
> real panel first. My honest read: Modal as the default, Docker/CPU as a
> fallback that now actually works rather than one that's theoretical.

---

## (C) The slow first load

> One more thing, unrelated to our stuff — PrimerCast takes about 21 seconds
> to load for me on a fresh browser and about 3 on a reload, and I had my
> agent dig into why since we'd inherit the same problem. It's not your app
> code. It's that Cloud Run is scaled to zero, so the first visit after any
> idle period has to spin up the whole container — mount the image, start
> Python, import torch, boot Streamlit — before it sends a single byte. The
> page isn't slow to load so much as there's no server there yet to load it
> from.
>
> Two fixes, cheapest first. A Cloud Scheduler job pinging `/_stcore/health`
> every five minutes keeps an instance alive for basically nothing, or
> `min_instances = 1` does it deterministically but you pay for an always-on
> container. We also found the frontend bundle ships uncompressed (3 MB where
> gzip gives 730 KB, and Google Frontend passes it straight through) — that's
> real but it's the smaller effect, worth under a second on a decent
> connection, so I'd do the keep-warm first and only bother with compression
> if it still feels slow. Streamlit actually bypasses gzip on purpose and
> expects a proxy or CDN to do it, so it's a deployment gap rather than a bug.
> There's a working nginx config in our repo if it's ever worth it.
>
> The other thing we did, which might interest you more: for the Modal-backed
> portal the container doesn't need torch at all, since the compute is
> elsewhere. Dropping it takes the image from 3.4 GB to a fraction of that,
> which attacks the cold start at the source instead of paying to paper over
> it. If you end up hosting the thin-client version you'd get that for free.
