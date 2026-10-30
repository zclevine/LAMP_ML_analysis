"""ML-guided LAMP primer set design.

The names re-exported here are the engine's public surface: what the Streamlit
app, the Modal harness and PrimerCast integrate against. They are re-exported
deliberately so a front end imports ``lamp_design.predict_tp`` rather than
reaching into ``lamp_design.models.bridge`` -- internal module layout can then
move without breaking callers.

Everything is imported lazily through ``__getattr__``. Importing
``lamp_design`` therefore does not import torch, which keeps ``lamp-design
prepare`` and the CLI's ``--help`` fast and lets the package be imported in an
environment with no model dependencies at all.

Typical use::

    from lamp_design import design, prepare_panel, read_fasta

    panel = prepare_panel(read_fasta("targets.fa")).panel
    result = design(panel, weights="/path/to/checkpoints", stride=1)

:func:`design` is the one-call entry point and is what every front end should
use; the stage-level functions below are exposed for callers that need to
interleave their own work (the Modal fan-out splits thermo from inference).
"""

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

#: Public name -> the module it actually lives in. One table, so there is a
#: single place to look when a front end asks "where does this come from".
_EXPORTS = {
    # orchestration
    "design": "lamp_design.commands.design",
    "build_conditions": "lamp_design.commands.design",
    "load_input_panel": "lamp_design.commands.design",
    "resolve_device": "lamp_design.commands.design",
    # input
    "prepare_panel": "lamp_design.commands.prepare",
    "load_panel": "lamp_design.commands.prepare",
    "write_panel": "lamp_design.commands.prepare",
    "PreparedPanel": "lamp_design.commands.prepare",
    "InputError": "lamp_design.commands.prepare",
    "read_fasta": "lamp_design.utils.sequences",
    "write_fasta": "lamp_design.utils.sequences",
    # generation
    "generate_candidates": "lamp_design.generation",
    "default_geometries": "lamp_design.generation",
    "canonical_geometry": "lamp_design.generation",
    "random_geometries": "lamp_design.generation",
    "count_candidate_slots": "lamp_design.generation",
    "Geometry": "lamp_design.generation",
    "DEFAULT_STRIDE": "lamp_design.generation",
    "MIN_DESIGNABLE_LENGTH": "lamp_design.generation",
    # thermodynamics
    "compute_thermo": "lamp_design.thermo",
    "compute_dg": "lamp_design.thermo",
    "compute_tm": "lamp_design.thermo",
    "WindowResolver": "lamp_design.thermo",
    # model
    "predict_tp": "lamp_design.models.bridge",
    "encode_pairs": "lamp_design.models.bridge",
    "mismatch_map": "lamp_design.models.bridge",
    "LyraEnsemble": "lamp_design.models.inference",
    "WeightsNotFound": "lamp_design.models.inference",
    "ENSEMBLE_SEEDS": "lamp_design.models.inference",
    # report
    "build_design_result": "lamp_design.report",
    "score_sets": "lamp_design.report",
    "select_diverse": "lamp_design.report",
    "result_to_dict": "lamp_design.report",
    "write_outputs": "lamp_design.report",
    "format_summary": "lamp_design.report",
    # contracts
    "Condition": "lamp_design.types",
    "DesignResult": "lamp_design.types",
    "Prediction": "lamp_design.types",
    "Primer": "lamp_design.types",
    "PrimerSet": "lamp_design.types",
    "ScoredSet": "lamp_design.types",
    "Target": "lamp_design.types",
    "TargetPanel": "lamp_design.types",
    "ThermoTable": "lamp_design.types",
    "MASTER_MIXES": "lamp_design.types",
    "PRIMER_TYPE_ORDER": "lamp_design.types",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    """Import on first access, so ``import lamp_design`` stays torch-free."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'lamp_design' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache, so later lookups skip __getattr__
    return value


def __dir__() -> list:
    return list(__all__)


if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from lamp_design.commands.design import (
        build_conditions,
        design,
        load_input_panel,
        resolve_device,
    )
    from lamp_design.commands.prepare import (
        InputError,
        PreparedPanel,
        load_panel,
        prepare_panel,
        write_panel,
    )
    from lamp_design.generation import (
        DEFAULT_STRIDE,
        MIN_DESIGNABLE_LENGTH,
        Geometry,
        canonical_geometry,
        count_candidate_slots,
        default_geometries,
        generate_candidates,
        random_geometries,
    )
    from lamp_design.models.bridge import encode_pairs, mismatch_map, predict_tp
    from lamp_design.models.inference import (
        ENSEMBLE_SEEDS,
        LyraEnsemble,
        WeightsNotFound,
    )
    from lamp_design.report import (
        build_design_result,
        format_summary,
        result_to_dict,
        score_sets,
        select_diverse,
        write_outputs,
    )
    from lamp_design.thermo import (
        WindowResolver,
        compute_dg,
        compute_thermo,
        compute_tm,
    )
    from lamp_design.types import (
        MASTER_MIXES,
        PRIMER_TYPE_ORDER,
        Condition,
        DesignResult,
        Prediction,
        Primer,
        PrimerSet,
        ScoredSet,
        Target,
        TargetPanel,
        ThermoTable,
    )
    from lamp_design.utils.sequences import read_fasta, write_fasta
