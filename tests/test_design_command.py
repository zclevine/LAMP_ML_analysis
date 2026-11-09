"""``lamp-design design`` -- conditions, input sniffing, and a real CLI run.

The pipeline stages have their own tests; this covers the orchestration that
only exists in ``commands/design.py``, and runs the actual CLI once end to end
so the wiring is exercised rather than asserted about.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lamp_design.commands.design import (
    DEFAULT_CONCENTRATIONS,
    DEFAULT_MASTER_MIXES,
    _ranking_condition,
    build_conditions,
    design,
    load_input_panel,
    resolve_device,
)
from lamp_design.generation import DEFAULT_N_SAMPLED, default_geometries
from lamp_design.commands.prepare import InputError
from lamp_design.types import Condition

rng = np.random.default_rng(99)


def _fasta(tmp_path, n: int = 3, length: int = 700, name: str = "targets.fa"):
    """A small panel of related sequences, written as FASTA."""
    base = "".join(rng.choice(list("ACGT"), size=length))
    lines = []
    for i in range(n):
        arr = np.array(list(base))
        if i:
            pos = rng.choice(length, size=max(1, length // 100), replace=False)
            arr[pos] = rng.choice(list("ACGT"), size=len(pos))
        lines.append(f">target_{i}\n{''.join(arr)}")
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


from _weights import WEIGHTS_MISSING_REASON, weights_or_none


def _weights():
    return weights_or_none()


requires_weights = pytest.mark.skipif(_weights() is None, reason=WEIGHTS_MISSING_REASON)


# --------------------------------------------------------------- conditions


def test_default_conditions_are_one_mix_at_two_concentrations():
    """1 mix x 2 concentrations, deliberately -- not the 2x2 it used to be.

    Ranking consumes exactly one condition, so extra *mixes* cost inference
    (89% of runtime) and change no ranking. The second *concentration* is kept
    because it is the robustness column, and it is cheap relative to a mix.
    """
    conds = build_conditions()
    assert len(conds) == len(DEFAULT_MASTER_MIXES) * len(DEFAULT_CONCENTRATIONS) == 2
    assert [c.master_mix for c in conds] == ["BST", "BST"]
    assert {c.target_conc for c in conds} == set(DEFAULT_CONCENTRATIONS)
    # The default ranking condition is the comfortable concentration, so the
    # low-copy prediction never silently drives the ranking.
    assert _ranking_condition(conds, None, None).target_conc == max(DEFAULT_CONCENTRATIONS)


def test_unknown_master_mix_is_rejected_with_the_trained_set():
    with pytest.raises(InputError) as exc:
        build_conditions(["BST", "TAQ"], [1000.0])
    assert "TAQ" in str(exc.value) and "BST" in str(exc.value)


@pytest.mark.parametrize("concs", [[0.0], [-5.0], []])
def test_bad_concentrations_are_rejected(concs):
    with pytest.raises(InputError):
        build_conditions(["BST"], concs)


def test_default_ranking_condition_is_first_mix_at_highest_conc():
    conds = build_conditions(["SS", "BST"], [100.0, 10000.0])
    assert _ranking_condition(conds, None, None) == Condition("SS", 10000.0)


def test_explicit_ranking_condition_must_be_one_being_scored():
    conds = build_conditions(["BST"], [100.0])
    assert _ranking_condition(conds, "BST", 100.0) == Condition("BST", 100.0)
    with pytest.raises(InputError) as exc:
        _ranking_condition(conds, "SS", 100.0)
    # The message has to say what *is* available, or it is unactionable.
    assert "BST@100" in str(exc.value)


def test_resolve_device_passes_explicit_values_through():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") in {"cpu", "cuda"}


# --------------------------------------------------------------- input load


def test_loads_a_fasta(tmp_path):
    panel, prepared = load_input_panel(_fasta(tmp_path))
    assert panel.n_targets == 3
    assert prepared is not None, "preparing a FASTA must return its warnings/exclusions"


def test_loads_a_prepared_panel_json(tmp_path):
    from lamp_design.commands.prepare import prepare_panel, write_panel
    from lamp_design.utils.sequences import read_fasta

    prepared = prepare_panel(read_fasta(_fasta(tmp_path)))
    out = tmp_path / "panel.json"
    write_panel(prepared, out)

    panel, again = load_input_panel(out)
    assert panel.n_targets == 3
    assert again is None, "a panel JSON is already QC'd; it must not be re-prepared"
    assert panel.consensus == prepared.panel.consensus


def test_missing_input_is_a_user_error(tmp_path):
    with pytest.raises(InputError):
        load_input_panel(tmp_path / "nope.fa")


def test_neither_fasta_nor_panel_is_rejected_by_content_not_extension(tmp_path):
    """A mislabelled file gets a message about its content, not its name."""
    path = tmp_path / "targets.fa"
    path.write_text("ACGTACGTACGT\n")  # no '>' header
    with pytest.raises(InputError) as exc:
        load_input_panel(path)
    assert "FASTA" in str(exc.value)


def test_json_that_is_not_a_panel_says_so(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"something": "else"}')
    with pytest.raises(InputError) as exc:
        load_input_panel(path)
    assert "prepare" in str(exc.value)


# ------------------------------------------------------------------ pipeline


@requires_weights
def test_design_records_its_own_parameters(tmp_path):
    """``DesignResult.params`` is what lets outputs label their own columns."""
    panel, _ = load_input_panel(_fasta(tmp_path, n=2, length=600))
    conds = build_conditions(["BST"], [10000.0])
    result = design(
        panel,
        weights=_weights(),
        conditions=conds,
        stride=25,
        n=5,
        device="cpu",
    )
    assert result.scored_sets
    p = result.params
    assert p["stride"] == 25
    assert p["conditions"] == ["BST@10000"]
    assert p["ranking_condition"] == "BST@10000"
    assert p["n_candidates"] > 0
    assert p["n_predictions"] == p["n_pairs"] * len(conds)
    assert set(p["timings_s"]) >= {"generation_s", "thermo_s", "inference_s", "report_s"}
    # Geometry *ids*, not dataclass reprs. This regressed once: Geometry has
    # no `name`, so a `getattr(g, "name", str(g))` here wrote the whole repr
    # into every report -- readable, so nothing failed, but useless as a label.
    # The shipped default: the canonical layout plus DEFAULT_N_SAMPLED drawn
    # per anchor. Recorded by name so a result says which layouts produced it.
    assert p["geometries"] == ["canonical"] + [
        f"sampled-{i}" for i in range(DEFAULT_N_SAMPLED)
    ]
    assert all("Geometry(" not in g for g in p["geometries"])
    # JSON-serializable by construction -- it gets written to the report.
    json.dumps(p)


@requires_weights
def test_progress_callback_reports_every_stage(tmp_path):
    panel, _ = load_input_panel(_fasta(tmp_path, n=2, length=600))
    seen = []
    design(
        panel,
        weights=_weights(),
        conditions=build_conditions(["BST"], [10000.0]),
        stride=40,
        n=3,
        device="cpu",
        progress=lambda stage, message: seen.append(stage),
    )
    assert seen == ["generation", "thermo", "inference", "report"]


@requires_weights
def test_cli_design_writes_all_four_outputs(tmp_path, capsys):
    """The real CLI, on a real FASTA, to real files."""
    from lamp_design.cli import main

    fasta = _fasta(tmp_path, n=3, length=650)
    prefix = tmp_path / "run"
    main([
        "design",
        "--in", str(fasta),
        "--out", str(prefix),
        "--top", "5",
        "--stride", "20",
        "--master-mix", "BST",
        "--conc", "10000",
        "--device", "cpu",
        "--weights", _weights(),
    ])

    for suffix in (".sets.csv", ".per_target.csv", ".json", ".summary.txt"):
        path = tmp_path / f"run{suffix}"
        assert path.exists(), f"{suffix} was not written"
        assert path.stat().st_size > 0, f"{suffix} is empty"

    report = json.loads((tmp_path / "run.json").read_text())
    assert report["params"]["stride"] == 20
    assert report["params"]["device"] == "cpu"

    out = capsys.readouterr().out
    assert "generation" in out and "inference" in out
    assert "Ranked under BST@10000" in out


@requires_weights
def test_cli_design_accepts_a_prepared_panel(tmp_path):
    """prepare -> design, the two-step path the portal will use."""
    from lamp_design.cli import main

    fasta = _fasta(tmp_path, n=2, length=600)
    panel_json = tmp_path / "panel.json"
    main(["prepare", "--in", str(fasta), "--out", str(panel_json)])
    assert panel_json.exists()

    main([
        "design",
        "--in", str(panel_json),
        "--out", str(tmp_path / "run2"),
        "--top", "3",
        "--stride", "30",
        "--device", "cpu",
        "--weights", _weights(),
    ])
    assert (tmp_path / "run2.sets.csv").exists()
