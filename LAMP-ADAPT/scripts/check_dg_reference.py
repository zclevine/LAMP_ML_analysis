"""Assert that this environment's RNAduplex returns the validated dG values.

Run at image build time (see ``Dockerfile``) and in CI. Exits non-zero with the
offending pairs if any value drifts.

Why this exists instead of a version pin
----------------------------------------
Binding dG is a model *input*: every prediction is conditioned on it, so an
environment whose RNAduplex computes dG differently produces different designs
from the same input -- and it does so **silently**, because a different-but-valid
dG is not an error. ``deploy/modal/app.py`` guards this by pinning the binary to
``2.4.17`` and asserting the version at build time.

That pin is unavailable on the CLI image's base (bioconda ships no 2.4.17 build
for python 3.10), so this checks the property the pin was a proxy for. It is
the stronger check of the two: a version string would also pass on a 2.4.17
that had been patched, rebuilt against different parameters, or miscompiled.

The reference values
--------------------
Measured on the development host, ViennaRNA **2.4.17**, which is the version
the physics features were validated against -- all four features agreed with
the training CSVs at ``max|diff| = 0.00000``, and dG specifically matched
training ground truth on 196/196 spot-checked pairs across all three backends.

Tolerance is 1e-4 kcal/mol. RNAduplex prints two decimals, so anything within
that is formatting, and a real parameter-set change moves dG by whole
kcal/mol -- far outside it.
"""

from __future__ import annotations

import sys

#: Six fixed pairs: perfect match, GC-rich, AT-rich, one internal mismatch,
#: a mixed-composition primer, and an offset duplex. Chosen to touch different
#: regions of the nearest-neighbour parameter table, so a parameter change has
#: somewhere to show up.
PAIRS = [
    ("ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGT"),
    ("GGCCGGCCGGCCGGCCGGCC", "GGCCGGCCGGCCGGCCGGCC"),
    ("AATTAATTAATTAATTAATT", "AATTAATTAATTAATTAATT"),
    ("ACGTACGTACGTACGTACGT", "ACGTACGTACGTTCGTACGT"),
    ("TTGCAGTCCAGGTACTGCAT", "TTGCAGTCCAGGTACTGCAT"),
    ("GCATGCATGCATGCATGCAT", "ATGCATGCATGCATGCATGC"),
]

#: kcal/mol, from ViennaRNA 2.4.17 on the development host.
DG_REFERENCE = [-26.4, -36.799999, -15.9, -23.5, -26.4, -27.799999]

TOLERANCE = 1e-4


def main() -> int:
    from lamp_design.thermo import compute_dg

    observed = compute_dg(PAIRS)
    if len(observed) != len(DG_REFERENCE):
        print(
            f"compute_dg returned {len(observed)} values for {len(PAIRS)} pairs",
            file=sys.stderr,
        )
        return 1

    bad = []
    for (primer, target), got, want in zip(PAIRS, observed, DG_REFERENCE):
        if abs(float(got) - want) > TOLERANCE:
            bad.append((primer, target, float(got), want))

    if bad:
        print(
            "RNAduplex dG does not match the validated reference. Every "
            "prediction is conditioned on these values, so this environment "
            "would produce different designs than the ones the model was "
            "validated on -- without raising an error anywhere else.",
            file=sys.stderr,
        )
        for primer, target, got, want in bad:
            print(
                f"  {primer} x {target}: got {got:.6f}, expected {want:.6f} "
                f"(diff {got - want:+.6f})",
                file=sys.stderr,
            )
        print(
            "\nCheck the installed ViennaRNA. The reference is 2.4.17; see "
            "scripts/check_dg_reference.py and deploy/modal/app.py.",
            file=sys.stderr,
        )
        return 1

    print(f"dG reference OK: {len(PAIRS)} pairs within {TOLERANCE} kcal/mol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
