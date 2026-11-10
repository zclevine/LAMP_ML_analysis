# CPU-only, amd64 image for the LAMP design engine.
#
# Mirrors the structure of PrimerCast's Dockerfile (micromamba base, deps in
# one solve, package installed in its own layer, tools verified at build time),
# reduced to the single CPU variant this engine needs:
#
#   * RNAduplex (ViennaRNA) on PATH -- binding dG for every (primer, target)
#     pair. It is the pipeline bottleneck, not the model.
#   * MAFFT on PATH -- optional, used by `prepare` only when the input is an
#     unaligned multi-FASTA with sequences of differing length.
#   * CPU-only torch -- `cpuonly` goes in the SAME solve as pytorch so the
#     CUDA libraries are never written to a layer; removing them in a later
#     layer would not shrink the image.
#
# Model weights ARE baked in: the five-seed ensemble lives inside the package
# at src/lamp_design/weights/ and is 2.8 MB, so it costs nothing to carry and
# means this image needs no weights mount, no download and no environment
# variable. The build verifies below that all five seeds survived packaging.
#
# Build:
#   docker build -t lamp-design:cpu .
#
# Run. One mount, and `--user`: the image runs as a non-root user, so without
# it every write to /data fails with "Permission denied" *after* the pipeline
# has done all its work. Mapping to the caller's uid also means the outputs
# are owned by them, not by root.
#
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data \
#     lamp-design:cpu \
#     lamp-design prepare --in /data/targets.fa --out /data/panel.json
#
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data \
#     lamp-design:cpu \
#     lamp-design design --in /data/panel.json --out /data/sets
#
# Verified 2026-11-01: a stride-60 design on examples/H5N1_HA_demo.fasta runs
# to completion in-container in 15.4 s and writes all four outputs.
#
# To run a *different* ensemble -- a retrain, an ablation -- mount it and say
# so; a configured path that does not exist raises rather than falling back:
#   -v /host/ckpts:/mnt/ckpts:ro -e LAMP_DESIGN_WEIGHTS=/mnt/ckpts
# Either way `--weights` on the command line still wins. Such a mount must be
# a *dereferenced* copy: in a training directory the .pt files are often
# symlinks, and a bind mount of those gives the container dangling links.

FROM --platform=linux/amd64 mambaorg/micromamba:2.5.0-debian12-slim

LABEL org.opencontainers.image.description="ML-guided LAMP primer set design (CPU)"

# Layer 1: conda environment. Everything heavy -- the external binaries and
# the compiled Python deps -- is solved once here, so this layer is cached
# until the dependency list itself changes.
#
# Channel order matches PrimerCast: pytorch (for `cpuonly`), bioconda (mafft,
# viennarna, primer3-py), conda-forge (everything else).
#
# primer3-py is held below 2.1 deliberately: 2.1.0 changed the santalucia salt
# correction, which shifts every primer Tm by +2.17 C away from the values the
# model was trained on. The bound must match pyproject.toml -- see the Tm
# parity test in tests/test_thermo.py.
#: ViennaRNA is NOT pinned to a version here, and that needs explaining,
#: because deploy/modal/app.py pins the RNAduplex binary to 2.4.17 and says the
#: pin is load-bearing -- the installed RNAduplex decides the binding dG every
#: prediction is conditioned on, and a wrong one does not crash, it quietly
#: returns different numbers.
#:
#: bioconda has no 2.4.17 build for python 3.10 (its newest is py3.9), so the
#: pin is simply not available on this base. Rather than drop to 3.9 or build
#: ViennaRNA from source, the image asserts the thing the version pin was ever
#: a proxy for: that RNAduplex returns the dG values the features were
#: validated against. See the DG_REFERENCE check in the verify step below,
#: which is a stronger guarantee than a version string -- it would also catch a
#: 2.4.17 build that was patched or miscompiled.
#:
#: libopenblas is explicit because the `RNA` Python bindings link against it
#: and conda does not pull it in as a dependency of viennarna. Without it the
#: binary works, `import RNA` raises
#: "libopenblas.so.0: cannot open shared object file", and the image fails the
#: verify step -- which is how this was found. The Dockerfile had never been
#: built before CI was added.
RUN micromamba install -y -n base \
        -c pytorch -c bioconda -c conda-forge \
        python=3.10 \
        pytorch cpuonly \
        viennarna \
        libopenblas \
        mafft \
        "primer3-py>=2.0,<2.1" \
        biopython \
        numpy \
        pandas \
        scipy \
        einops \
    && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH="/opt/conda/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1

# Layer 2: install the package. `--no-deps` on purpose: conda above owns every
# dependency, and letting pip re-resolve them would pull the PyPI `ViennaRNA`
# sdist (a from-source build of a library we already have) and the CUDA torch
# wheel on top of the CPU one. The import check below is what guards against a
# dependency that conda is missing.
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml /app/
COPY --chown=$MAMBA_USER:$MAMBA_USER src/ /app/src/
WORKDIR /app
RUN pip install --no-cache-dir --no-deps .

# Layer 2b: the Fusion model, on by default.
#
# Fusion is the four-model blend used by `lamp-design evaluate --model fusion`.
# It is NOT reachable from `design` -- see models/fusion.py for the measured
# reason (~29x slower per prediction; a design run projects to ~1.9 h on GPU).
#
# TabPFN's 43 MB checkpoint is DOWNLOADED HERE, at build time, rather than
# committed to the repo. The resulting image needs no network to run, which
# is the property that matters for an air-gapped or egress-restricted
# deployment; it just needs network to build. TABPFN_MODEL_CACHE_DIR is what
# makes that work without guessing at $HOME.
#
# ModelVersion.V2 is deliberate and load-bearing: TabPFN 2.5 and newer are
# released under NON-COMMERCIAL licences, while V2 is Apache 2.0 plus an
# attribution clause. Shipping the weights carries that attribution
# obligation -- include the licence and display "Built with
# PriorLabs-TabPFN".
#
# Build without it:  docker build --build-arg WITH_FUSION=0 -t lamp-design:cpu .
ARG WITH_FUSION=1
ENV TABPFN_MODEL_CACHE_DIR=/opt/tabpfn
ENV WITH_FUSION_BUILT=${WITH_FUSION}
USER root
RUN mkdir -p /opt/tabpfn && chown $MAMBA_USER /opt/tabpfn
USER $MAMBA_USER
RUN if [ "$WITH_FUSION" = "1" ]; then \
      pip install --no-cache-dir "catboost==1.2.10" "xgboost==3.2.0" "tabpfn==8.5.0" \
      && python -c "\
from pathlib import Path;\
from tabpfn.constants import ModelVersion;\
from tabpfn.model_loading import download_model;\
dest = Path('/opt/tabpfn/tabpfn-v2-regressor.ckpt');\
r = download_model(dest, version=ModelVersion.V2, which='regressor');\
assert r == 'ok', r;\
print('tabpfn v2 regressor baked:', dest.stat().st_size // 1048576, 'MB')"; \
    else echo "WITH_FUSION=0: skipping the Fusion dependencies"; fi

# Layer 3: the rest of the repo (docs, tests, deploy scaffolding).
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# Verify the external tools, the CPU-only torch, and the CLI surface.
#
# The model-path import on the last line is not redundant with the one above
# it. `lamp-design --help` deliberately does NOT import torch (model imports
# are function-local so `--help` stays fast), and the flat import list only
# covers direct third-party deps -- so neither touches models/lyra/s4d.py.
# A missing `einops` therefore built a clean image that failed at first
# inference, which is exactly what happened on Modal. Importing the real
# entry point is the only check that closes that gap.
RUN lamp-design --help > /dev/null \
    && lamp-design prepare --help > /dev/null \
    && RNAduplex --version \
    && mafft --version 2>&1 | grep -qiE '^v[0-9]' \
    && python -c "import numpy, pandas, Bio, primer3, RNA, scipy, torch" \
    && python -c "import torch, sys; sys.exit(0 if torch.version.cuda is None else 'torch is not the CPU build')" \
    && python -c "from lamp_design.models.bridge import predict_tp; print('model path imports')" \
    && python -c "\
from lamp_design.models.inference import resolve_weights_dir, discover_checkpoints;\
d = resolve_weights_dir();\
c = discover_checkpoints(d);\
assert sorted(c) == [0, 1, 2, 3, 42], f'expected 5 seeds, found {sorted(c)}';\
print('bundled weights:', len(c), 'seeds at', d)" \
    && python /app/scripts/check_dg_reference.py \
    && python -c "\
import os, sys;\
sys.exit(0) if os.environ.get('WITH_FUSION_BUILT') != '1' else None;\
from lamp_design.models.fusion import load_fusion;\
f = load_fusion(device='cpu');\
print('fusion bundle resolves:', sorted(f.weights))"

# No LAMP_DESIGN_WEIGHTS here on purpose. Leaving it unset lets the bundled
# ensemble be the default; presetting it to a mount point that may not exist
# would turn "docker run" into an error for the common case. Override it only
# to run a different ensemble.

WORKDIR /data
CMD ["lamp-design", "--help"]
