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
# Model weights are NOT baked in and never will be (docs/ARCHITECTURE.md).
# They are mounted at run time and found via $LAMP_DESIGN_WEIGHTS, which this
# image presets to /weights so the mount point is self-documenting.
#
# Build:
#   docker build -t lamp-design:cpu .
#
# Run (note the two mounts: read-only weights, read-write data):
#   docker run --rm \
#     -v /host/path/LAMP_experiment_20260905_021750_tp60_A_v0t9:/weights:ro \
#     -v "$PWD":/data \
#     lamp-design:cpu \
#     lamp-design prepare --in /data/targets.fa --out /data/panel.json
#
#   docker run --rm \
#     -v /host/path/checkpoints:/weights:ro -v "$PWD":/data \
#     lamp-design:cpu \
#     lamp-design design --in /data/panel.json --out /data/sets.tsv
#
# To put the checkpoints somewhere else, override the variable and the mount:
#   -v /host/ckpts:/mnt/ckpts:ro -e LAMP_DESIGN_WEIGHTS=/mnt/ckpts
# Either way `--weights` on the command line still wins.

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
RUN micromamba install -y -n base \
        -c pytorch -c bioconda -c conda-forge \
        python=3.10 \
        pytorch cpuonly \
        viennarna \
        mafft \
        "primer3-py>=2.0,<2.1" \
        biopython \
        numpy \
        pandas \
        scipy \
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

# Layer 3: the rest of the repo (docs, tests, deploy scaffolding).
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# Verify the external tools, the CPU-only torch, and the CLI surface.
RUN lamp-design --help > /dev/null \
    && lamp-design prepare --help > /dev/null \
    && RNAduplex --version \
    && mafft --version 2>&1 | grep -qiE '^v[0-9]' \
    && python -c "import numpy, pandas, Bio, primer3, RNA, scipy, torch" \
    && python -c "import torch, sys; sys.exit(0 if torch.version.cuda is None else 'torch is not the CPU build')"

# Weights are mounted here; nothing is baked in.
ENV LAMP_DESIGN_WEIGHTS=/weights

WORKDIR /data
CMD ["lamp-design", "--help"]
