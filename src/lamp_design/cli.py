#!/usr/bin/env python
"""lamp-design - ML-guided LAMP primer set design CLI.

Single entry point, argparse subcommands, one module per command exposing
``register(subparsers)`` -- the same shape as ``primercast.cli``, because this
engine gets handed to that codebase's maintainer for portal integration.

All three commands are implemented: ``prepare`` (align and QC an input
panel), ``design`` (search for new primer sets) and ``evaluate`` (score sets
that already exist). Each lives in its own module under ``commands/`` and is
discovered by name, so adding a fourth means adding a module and one line in
:func:`build_parser`, not restructuring this one.

Model weights are never bundled (see ``docs/ARCHITECTURE.md``). They are
resolved from ``--weights`` or ``$LAMP_DESIGN_WEIGHTS`` by
:func:`resolve_weights`, which is the one place that policy lives.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

from lamp_design import __version__

#: Environment variable holding the trained-checkpoint directory.
WEIGHTS_ENV_VAR = "LAMP_DESIGN_WEIGHTS"

#: No subcommand given.
EXIT_USAGE = 1
#: The user has to change something about the input (same code argparse uses).
EXIT_INPUT_ERROR = 2

__all__ = [
    "WEIGHTS_ENV_VAR",
    "WeightsError",
    "add_weights_argument",
    "resolve_weights",
    "build_parser",
    "main",
]


class WeightsError(ValueError):
    """The model weights could not be resolved."""


# ---------------------------------------------------------------------------
# Weights resolution -- never bundled, always resolved at run time
# ---------------------------------------------------------------------------


def add_weights_argument(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--weights`` option to a subcommand parser."""
    parser.add_argument(
        "--weights",
        default=None,
        help="Trained checkpoint directory (holding seed_*/.../best_model_*.pt). "
             f"Defaults to ${WEIGHTS_ENV_VAR}. Weights are never bundled with this "
             "package.",
    )


def resolve_weights(
    explicit: Optional[str | Path] = None,
    *,
    env: Optional[dict] = None,
    required: bool = True,
) -> Optional[Path]:
    """Resolve the model-weights path from ``--weights`` or the environment.

    Args:
        explicit: Value of ``--weights``, if given. Takes precedence.
        env: Environment mapping to read ``LAMP_DESIGN_WEIGHTS`` from
            (defaults to ``os.environ``; injectable for tests).
        required: When False, return ``None`` instead of raising if nothing is
            configured. A path that *is* configured but does not exist is
            always an error -- that is a typo, not an absent setting.

    Returns:
        The resolved path, or ``None`` when nothing is configured and
        ``required`` is False.

    Raises:
        WeightsError: nothing configured (and ``required``), or the configured
            path does not exist.
    """
    env = os.environ if env is None else env

    source, raw = "--weights", explicit
    if raw is None or raw == "":
        source, raw = f"${WEIGHTS_ENV_VAR}", env.get(WEIGHTS_ENV_VAR)

    if raw is None or raw == "":
        if not required:
            return None
        raise WeightsError(
            "model weights are not configured. The trained Simplified Lyra "
            "checkpoints are deliberately never bundled with this package, so they "
            "have to be pointed at: pass --weights /path/to/checkpoints or set "
            f"{WEIGHTS_ENV_VAR}=/path/to/checkpoints. That directory holds the "
            "five-seed ensemble as seed_*/.../best_model_*.pt (see "
            "docs/ARCHITECTURE.md)."
        )

    path = Path(raw).expanduser()
    if not path.exists():
        raise WeightsError(
            f"model weights path '{path}' does not exist (from {source}). Check the "
            "path, or mount the checkpoint directory if you are running in a "
            f"container (the image expects it at ${WEIGHTS_ENV_VAR})."
        )
    return path


def _register_command(subparsers, name: str) -> None:
    """Register ``lamp_design.commands.<name>``'s subcommand."""
    module = importlib.import_module(f"lamp_design.commands.{name}")
    module.register(subparsers)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The full argument parser, with every subcommand registered."""
    parser = argparse.ArgumentParser(
        prog="lamp-design",
        description="ML-guided LAMP primer set design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  lamp-design prepare --in targets.fa --out panel.json
  lamp-design prepare --in alignment.fa --pre-aligned --consensus-out consensus.fa
  lamp-design design --in panel.json --out sets.tsv --weights /path/to/checkpoints
  lamp-design evaluate --in my_primers.tsv --targets panel.json --out scores.tsv

Model weights are never bundled: pass --weights or set {WEIGHTS_ENV_VAR}.

For more information on a specific command:
  lamp-design <command> --help
""",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available subcommands",
        metavar="<command>",
    )

    for name in ("prepare", "design", "evaluate"):
        _register_command(subparsers, name)

    return parser


def _user_errors() -> Tuple[type, ...]:
    """Exceptions that mean "the user has to change something", not a bug."""
    from lamp_design.commands.prepare import InputError
    from lamp_design.utils.sequences import SequenceValidationError

    return (InputError, SequenceValidationError, WeightsError, OSError)


def main(argv: Optional[list] = None) -> None:
    """Main entry point for the lamp-design CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(EXIT_USAGE)

    from lamp_design.utils.diagnostics import NO_CANDIDATES_EXIT_CODE, NoCandidatesError

    try:
        args.func(args)
    except NoCandidatesError as exc:
        print(f"\nNO CANDIDATES: {exc}\n", file=sys.stderr, flush=True)
        sys.exit(NO_CANDIDATES_EXIT_CODE)
    except _user_errors() as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        sys.exit(EXIT_INPUT_ERROR)


if __name__ == "__main__":
    main()
