"""Per-stage funnel diagnostics: which filter emptied the result.

The pipeline narrows candidates stage by stage (prepare -> generation ->
thermo -> encoding -> inference -> report). Each stage records a *funnel* --
how many candidates survive every filter it applies, with the params that
control that filter and a human-readable hint -- so an empty result can say
*which* filter emptied it instead of just "no primer sets found".

This is deliberately the same shape, and the same method names, as
``primercast.utils.diagnostics`` (``record`` / ``blocking_step`` / ``empty`` /
``to_dict`` / ``write`` / ``message`` / ``finish``, plus module-level
``diagnostics_path``, ``format_message`` and ``explain_run``), because this
engine gets handed to that codebase's maintainer for portal integration.

Two LAMP-specific additions:

* :meth:`StageDiagnostics.as_funnel` and :meth:`StageDiagnostics.apply_to`
  flatten the funnel into ``DesignResult.funnel``, which is the contract in
  ``types.py``.
* :meth:`StageDiagnostics.raise_if_empty` is the library-safe sibling of
  ``finish()``: it raises instead of calling ``sys.exit``, for use inside the
  engine (and in the GPU harness, where exiting the worker is not useful).

PrimerCast's ``append_step`` / ``run_warnings`` / rescue machinery is not
mirrored: it exists for that pipeline's rescue re-evaluation step, which has
no LAMP counterpart.

Depends only on the standard library.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SUFFIX = ".diagnostics.json"
#: Exit code used when a stage stops because a filter removed every candidate.
NO_CANDIDATES_EXIT_CODE = 3

#: Human-readable stage names, in pipeline order. The order is used to report
#: the earliest failing stage first.
STAGE_LABELS: Dict[str, str] = {
    "prepare": "Input preparation",
    "generation": "Candidate generation",
    "thermo": "Thermodynamic features",
    "encoding": "Tensor encoding",
    "inference": "Model inference",
    "report": "Ranking and output",
}


class NoCandidatesError(RuntimeError):
    """A stage that must produce candidates produced none.

    ``diagnostics`` is the :class:`StageDiagnostics` that explains why, so a
    caller can show the funnel rather than only the message.
    """

    def __init__(self, message: str, diagnostics: "StageDiagnostics" | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def diagnostics_path(output_path: str | Path) -> Path:
    """Diagnostics file belonging to a stage output."""
    return Path(f"{output_path}{SUFFIX}")


@dataclass
class StageDiagnostics:
    """Funnel of candidate counts for one pipeline stage.

    ``fatal`` marks stages whose output must be non-empty for the run to
    continue. ``unit`` names what is being counted ("primer sets", "target
    sequences", ...) so the message reads correctly for every stage.
    """

    stage: str
    target: str = ""
    fatal: bool = True
    unit: str = "candidates"
    steps: List[Dict[str, Any]] = field(default_factory=list)
    #: Extra stage-specific data saved alongside the funnel.
    details: Dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        filter_name: str,
        remaining: int,
        params: Dict[str, Any] | None = None,
        breakdown: Dict[str, int] | None = None,
        hint: str = "",
    ) -> int:
        """Record how many candidates remain after ``filter_name``.

        ``params`` are the knobs that control this filter -- they are what the
        user has to change, so they go in the message. ``breakdown`` splits the
        count (e.g. per primer type); ``remaining`` should then be the
        bottleneck. Returns ``remaining`` so calls can be used inline.
        """
        entry: Dict[str, Any] = {"filter": filter_name, "remaining": int(remaining)}
        if params:
            entry["params"] = {k: _plain(v) for k, v in params.items()}
        if breakdown:
            entry["breakdown"] = {k: int(v) for k, v in breakdown.items()}
        if hint:
            entry["hint"] = hint
        self.steps.append(entry)
        return int(remaining)

    def blocking_step(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        """``(index, step)`` of the first filter that left zero, if any."""
        for i, step in enumerate(self.steps):
            if step["remaining"] == 0:
                return i, step
        return None

    @property
    def empty(self) -> bool:
        return self.blocking_step() is not None

    @property
    def label(self) -> str:
        return STAGE_LABELS.get(self.stage, self.stage)

    def as_funnel(self, qualify: bool = True) -> Dict[str, int]:
        """Flatten to ``{step name: remaining}`` for ``DesignResult.funnel``.

        Keys are prefixed with the stage by default (``"generation: Tm / GC"``)
        so several stages can merge into the one dict ``types.py`` gives us
        without colliding, and so the report can group by stage.
        """
        prefix = f"{self.stage}: " if qualify else ""
        return {f"{prefix}{s['filter']}": int(s["remaining"]) for s in self.steps}

    def apply_to(self, result: Any) -> Any:
        """Merge this funnel into a ``DesignResult`` and return the result."""
        result.funnel.update(self.as_funnel())
        return result

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "stage": self.stage,
            "stage_label": self.label,
            "target": self.target,
            "fatal": self.fatal,
            "unit": self.unit,
            "empty": self.empty,
            "steps": self.steps,
        }
        if self.details:
            data["details"] = self.details
        return data

    def write(self, output_path: str | Path) -> Path:
        path = diagnostics_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def message(self) -> str:
        return format_message(self.to_dict())

    def raise_if_empty(self) -> None:
        """Raise :class:`NoCandidatesError` if a fatal stage ended up empty.

        The in-process version of :meth:`finish`, for library callers.
        """
        if self.fatal and self.empty:
            raise NoCandidatesError(self.message(), self)

    def finish(self, output_path: str | Path) -> None:
        """Write the diagnostics; stop the run if a fatal stage is empty.

        Exits with :data:`NO_CANDIDATES_EXIT_CODE` and prints the explanation
        to stderr, so it also shows up in a workflow log next to the step.
        """
        self.write(output_path)
        if self.fatal and self.empty:
            print(f"\nNO CANDIDATES: {self.message()}\n", file=sys.stderr, flush=True)
            sys.exit(NO_CANDIDATES_EXIT_CODE)


def merge_funnels(*diagnostics: StageDiagnostics) -> Dict[str, int]:
    """Combined ``DesignResult.funnel`` for several stages, in call order."""
    funnel: Dict[str, int] = {}
    for diag in diagnostics:
        funnel.update(diag.as_funnel())
    return funnel


def first_blocking(
    diagnostics: Iterable[StageDiagnostics],
) -> Optional[StageDiagnostics]:
    """The earliest fatal stage (in pipeline order) that ran out of candidates."""
    order = list(STAGE_LABELS)
    empty = [d for d in diagnostics if d.fatal and d.empty]
    if not empty:
        return None
    return min(
        empty,
        key=lambda d: order.index(d.stage) if d.stage in order else len(order),
    )


def format_message(data: Dict[str, Any]) -> str:
    """One-paragraph explanation of which filter emptied a stage."""
    label = data.get("stage_label") or STAGE_LABELS.get(data.get("stage", ""), "")
    target = data.get("target")
    where = f"{label} ({target})" if target else label
    unit = data.get("unit") or "candidates"
    steps = data.get("steps") or []

    for i, step in enumerate(steps):
        if step.get("remaining") != 0:
            continue
        before = steps[i - 1] if i > 0 else None
        if before is None:
            text = f"{where}: no {unit} to start with at '{step['filter']}'."
        else:
            text = (
                f"{where}: no {unit} left after the '{step['filter']}' filter "
                f"({before['remaining']:,} -> 0)."
            )
        if step.get("breakdown"):
            parts = ", ".join(f"{k} {v:,}" for k, v in step["breakdown"].items())
            text += f" Remaining by type: {parts}."
        if step.get("params"):
            params = ", ".join(f"{k}={v}" for k, v in step["params"].items())
            text += f" Controlled by: {params}."
        if step.get("hint"):
            text += f" {step['hint']}"
        return text
    return f"{where}: {unit} remained after every filter."


@dataclass
class Diagnosis:
    """A stage that ran out of candidates, as found in a run directory."""

    stage: str
    stage_label: str
    target: str
    fatal: bool
    message: str
    steps: List[Dict[str, Any]]
    path: Path


def explain_run(
    run_dir: str | Path,
    include_non_fatal: bool = False,
    since: float | None = None,
) -> List[Diagnosis]:
    """Find stages in a run that ran out of candidates, earliest stage first.

    ``since`` (a Unix timestamp) ignores files left by an earlier attempt of a
    re-run. Unreadable files are skipped: a missing explanation must never
    break the page that shows it.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    order = list(STAGE_LABELS)
    found: List[Diagnosis] = []
    for path in sorted(run_dir.rglob(f"*{SUFFIX}")):
        try:
            if since is not None and path.stat().st_mtime < since:
                continue
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not data.get("empty"):
            continue
        if not data.get("fatal", True) and not include_non_fatal:
            continue
        found.append(
            Diagnosis(
                stage=data.get("stage", ""),
                stage_label=data.get("stage_label", ""),
                target=data.get("target", ""),
                fatal=bool(data.get("fatal", True)),
                message=format_message(data),
                steps=data.get("steps") or [],
                path=path,
            )
        )
    found.sort(key=lambda d: order.index(d.stage) if d.stage in order else len(order))
    return found


def _plain(value: Any) -> Any:
    """JSON-friendly param value (numpy scalars, integral floats -> int)."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


__all__ = [
    "SUFFIX",
    "NO_CANDIDATES_EXIT_CODE",
    "STAGE_LABELS",
    "NoCandidatesError",
    "StageDiagnostics",
    "Diagnosis",
    "diagnostics_path",
    "merge_funnels",
    "first_blocking",
    "format_message",
    "explain_run",
]
