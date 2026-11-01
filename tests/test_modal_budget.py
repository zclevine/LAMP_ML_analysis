"""The Modal spend circuit breaker.

This is the control that stands between a runaway or abusive caller and the
account's credits, so it is tested directly rather than trusted. The logic is
pure arithmetic over a ledger, so the Modal ``Dict`` is replaced with an
in-memory stand-in and none of this touches the network or a GPU.

``deploy/modal/app.py`` is loaded by path because ``deploy/`` is not a package
-- the same approach ``tests/test_streamlit_app.py`` uses for the front end.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
APP_PATH = REPO / "deploy" / "modal" / "app.py"

HOUR = 3600.0
DAY = 86400.0


def _load_app():
    """Import the Modal harness. Modal objects are lazy, so this is offline."""
    spec = importlib.util.spec_from_file_location("lamp_modal_app_under_test", APP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDict:
    """Stands in for ``modal.Dict``: get/setitem, and optionally broken."""

    def __init__(self, broken: bool = False):
        self.store: dict = {}
        self.broken = broken
        self.writes = 0

    def get(self, key, default=None):
        if self.broken:
            raise RuntimeError("ledger unreachable")
        return self.store.get(key, default)

    def __setitem__(self, key, value):
        if self.broken:
            raise RuntimeError("ledger unreachable")
        self.writes += 1
        self.store[key] = value


@pytest.fixture
def app(monkeypatch):
    module = _load_app()
    monkeypatch.setattr(module, "budget_dict", FakeDict())
    monkeypatch.setattr(module, "BUDGET_ENABLED", True)
    return module


# ------------------------------------------------------------------ accounting


def test_an_empty_ledger_reports_zero(app):
    usage = app._budget_usage()
    assert usage["last_hour_gpu_hours"] == 0.0
    assert usage["last_day_gpu_hours"] == 0.0
    assert usage["total_gpu_hours"] == 0.0


def test_a_recorded_run_shows_up_in_all_three_windows(app):
    now = 1_000_000.0
    app._budget_record(HOUR, "run-a", now=now)
    usage = app._budget_usage(now=now)
    assert usage["last_hour_gpu_hours"] == pytest.approx(1.0)
    assert usage["last_day_gpu_hours"] == pytest.approx(1.0)
    assert usage["total_gpu_hours"] == pytest.approx(1.0)


def test_windows_are_trailing_not_calendar(app):
    """A run 90 minutes ago counts against the day but not the hour."""
    now = 1_000_000.0
    app._budget_record(2 * HOUR, "old", now=now - 90 * 60)
    app._budget_record(HOUR, "new", now=now)
    usage = app._budget_usage(now=now)
    assert usage["last_hour_gpu_hours"] == pytest.approx(1.0)
    assert usage["last_day_gpu_hours"] == pytest.approx(3.0)


def test_the_cumulative_total_survives_pruning(app):
    """The whole point of keeping `total_gpu_seconds` separately.

    Entries older than 24h are dropped from the log, so if the total were
    recomputed from the log it would silently reset every day -- turning a
    cumulative ceiling into a rolling one.
    """
    now = 1_000_000.0
    app._budget_record(5 * HOUR, "ancient", now=now - 3 * DAY)
    app._budget_record(HOUR, "recent", now=now)
    usage = app._budget_usage(now=now)
    assert usage["last_day_gpu_hours"] == pytest.approx(1.0), "old run should age out"
    assert usage["total_gpu_hours"] == pytest.approx(6.0), "total must not age out"


def test_zero_and_negative_are_not_recorded(app):
    app._budget_record(0.0, "zero")
    app._budget_record(-5.0, "negative")
    assert app._budget_usage()["total_gpu_hours"] == 0.0


def test_the_log_is_bounded(app):
    """A flood of tiny runs must not grow the Dict entry without limit."""
    now = 1_000_000.0
    for i in range(5200):
        app._budget_record(1.0, f"run-{i}", now=now)
    entry = app.budget_dict.store[app._BUDGET_KEY]
    assert len(entry["runs"]) == 5000
    # Pruning the log must not lose the total.
    assert entry["total_gpu_seconds"] == pytest.approx(5200.0)


# ---------------------------------------------------------------------- gating


def test_an_idle_account_allows_fanout(app):
    assert app._budget_gate("run-1") is True


def test_the_hourly_ceiling_withdraws_fanout_without_refusing(app, monkeypatch):
    """Degrade, do not fail: the job still completes on one container.

    The ceilings are set explicitly here rather than using the defaults,
    because with the shipped values (hourly 20, total 20) consuming enough to
    trip the hourly rule also trips the cumulative one, which refuses first --
    see `test_the_shipped_hourly_ceiling_is_unreachable`.
    """
    import time as _time

    monkeypatch.setattr(app, "BUDGET_HOURLY_GPU_HOURS", 2.0)
    monkeypatch.setattr(app, "BUDGET_DAILY_GPU_HOURS", 100.0)
    monkeypatch.setattr(app, "BUDGET_TOTAL_GPU_HOURS", 100.0)

    app.budget_dict.store[app._BUDGET_KEY] = {
        "runs": [[_time.time(), 3 * HOUR, "burst"]],
        "total_gpu_seconds": 3 * HOUR,
    }
    assert app._budget_gate("run-2") is False, "fan-out should be withdrawn"


def test_the_daily_ceiling_refuses_the_run(app):
    import time as _time

    app.budget_dict.store[app._BUDGET_KEY] = {
        "runs": [[_time.time(), app.BUDGET_DAILY_GPU_HOURS * HOUR, "heavy"]],
        "total_gpu_seconds": 0.0,
    }
    with pytest.raises(app.BudgetExceeded) as exc:
        app._budget_gate("run-3")
    message = str(exc.value)
    assert "24-hour" in message
    assert "LAMP_DESIGN_DAILY_GPU_HOURS" in message, "must name the knob to change"
    assert "trailing" in message, "must explain that capacity returns gradually"


def test_the_total_ceiling_refuses_and_says_it_does_not_reset(app):
    app.budget_dict.store[app._BUDGET_KEY] = {
        "runs": [],
        "total_gpu_seconds": app.BUDGET_TOTAL_GPU_HOURS * HOUR,
    }
    with pytest.raises(app.BudgetExceeded) as exc:
        app._budget_gate("run-4")
    message = str(exc.value)
    assert "cumulative" in message
    assert "does not reset" in message
    assert "LAMP_DESIGN_TOTAL_GPU_HOURS" in message


def test_the_total_ceiling_is_checked_before_the_daily_one(app):
    """Both tripped -> report the one that will not clear by waiting."""
    import time as _time

    app.budget_dict.store[app._BUDGET_KEY] = {
        "runs": [[_time.time(), app.BUDGET_DAILY_GPU_HOURS * HOUR, "heavy"]],
        "total_gpu_seconds": app.BUDGET_TOTAL_GPU_HOURS * HOUR,
    }
    with pytest.raises(app.BudgetExceeded) as exc:
        app._budget_gate("run-5")
    assert "cumulative" in str(exc.value)


def test_budget_exceeded_is_not_a_value_error(app):
    """A caller must be able to tell a bad request from a spent budget.

    `_validate_request` raises ValueError for malformed input; conflating the
    two would make "fix your FASTA" and "come back later" indistinguishable,
    and would map to the wrong HTTP status at the endpoint.
    """
    assert not issubclass(app.BudgetExceeded, ValueError)
    assert issubclass(app.BudgetExceeded, RuntimeError)


def test_disabling_the_budget_bypasses_every_ceiling(app, monkeypatch):
    monkeypatch.setattr(app, "BUDGET_ENABLED", False)
    app.budget_dict.store[app._BUDGET_KEY] = {
        "runs": [],
        "total_gpu_seconds": 10_000 * HOUR,
    }
    assert app._budget_gate("run-6") is True


# ------------------------------------------------------------------ resilience


def test_an_unreachable_ledger_does_not_fail_the_run(app, monkeypatch):
    """Fail open, loudly.

    A spend control that takes the service down when its bookkeeping is
    unavailable has converted a billing concern into an outage. The tradeoff
    is deliberate and is why Modal's own workspace spend limit should be set
    too -- that one cannot be bypassed by a bug here.
    """
    monkeypatch.setattr(app, "budget_dict", FakeDict(broken=True))
    assert app._budget_gate("run-7") is True
    app._budget_record(HOUR, "run-7")  # must not raise


def test_a_corrupt_ledger_entry_is_ignored_not_fatal(app):
    """A damaged ledger must not raise -- it reads as "no usage recorded".

    `total_gpu_seconds: "x"` used to raise ValueError straight out of
    `_budget_state`, which would have failed every design run until someone
    cleared the Dict by hand: a bookkeeping fault escalated into an outage.
    """
    app.budget_dict.store[app._BUDGET_KEY] = {"runs": "not a list", "total_gpu_seconds": "x"}
    usage = app._budget_usage()
    assert usage["last_day_gpu_hours"] == 0.0
    assert usage["total_gpu_hours"] == 0.0
    # And the breaker still answers rather than exploding.
    assert app._budget_gate("run-corrupt") is True


def test_malformed_log_rows_are_skipped(app):
    import time as _time

    app.budget_dict.store[app._BUDGET_KEY] = {
        "runs": [[_time.time(), HOUR, "good"], ["bad"], None, 42],
        "total_gpu_seconds": HOUR,
    }
    usage = app._budget_usage()
    assert usage["last_day_gpu_hours"] == pytest.approx(1.0)


# ------------------------------------------------------- consistency with code


def test_the_hourly_ceiling_is_documented_as_currently_unreachable(app):
    """On the free plan `max_containers` is the tighter control.

    10 concurrent GPUs cannot consume more than 10 GPU-hours in a wall-clock
    hour, so a 20 GPU-hour hourly ceiling cannot trip. If someone lowers the
    hourly ceiling below that physical limit it becomes live, and if someone
    raises max_containers it becomes live too -- either way the relationship
    is worth asserting so it is not discovered by surprise.
    """
    physical_max = float(app.INFER_MAX_CONTAINERS)
    if app.BUDGET_HOURLY_GPU_HOURS > physical_max:
        # Comments wrap mid-sentence AND carry `#:` / `#` prefixes, so both
        # have to come off before the phrase is one contiguous string.
        raw = APP_PATH.read_text()
        text = " ".join(
            line.lstrip().lstrip("#").lstrip(":").strip() for line in raw.splitlines()
        )
        text = " ".join(text.split())
        assert "cannot currently be reached" in text, (
            "the hourly ceiling exceeds what max_containers physically allows, "
            "so the comment explaining that must stay"
        )


def test_every_ceiling_is_env_overridable(app):
    """These are the knobs someone reaches for in a hurry."""
    text = APP_PATH.read_text()
    for name in (
        "LAMP_DESIGN_HOURLY_GPU_HOURS",
        "LAMP_DESIGN_DAILY_GPU_HOURS",
        "LAMP_DESIGN_TOTAL_GPU_HOURS",
        "LAMP_DESIGN_BUDGET",
    ):
        assert name in text, f"{name} is not readable from the environment"


def _function_body(text: str, signature: str) -> str:
    """Source of one top-level function, from its `def` to the next one.

    A fixed character window was the first attempt and it silently truncated
    `design_fanout` mid-docstring, so the assertion below passed or failed on
    comment length rather than on behaviour.
    """
    start = text.index(signature)
    # design_single_container is the last @app.function in the file, so there
    # is no following decorator to stop at -- run to the local entrypoint
    # banner, or to end of file.
    for marker in ("\n@app.function", "\n@app.local_entrypoint"):
        at = text.find(marker, start + 1)
        if at != -1:
            return text[start:at]
    return text[start:]


@pytest.mark.parametrize(
    "signature", ["def design_fanout", "def design_single_container"]
)
def test_both_gpu_paths_consult_the_breaker(app, signature):
    """A GPU path that skips the gate is a hole in the control."""
    body = _function_body(APP_PATH.read_text(), signature)
    assert "_budget_gate(" in body, f"{signature} does not call _budget_gate"


@pytest.mark.parametrize(
    "signature", ["def design_fanout", "def design_single_container"]
)
def test_both_gpu_paths_record_what_they_spent(app, signature):
    body = _function_body(APP_PATH.read_text(), signature)
    assert "_budget_record(" in body, f"{signature} does not record GPU seconds"


def test_the_shipped_hourly_ceiling_is_unreachable(app):
    """Documents a real consequence of the shipped defaults.

    Two independent things stop the hourly rule from ever firing:

    1. `INFER_MAX_CONTAINERS` = 10 caps consumption at 10 GPU-hours per
       wall-clock hour, below the 20 GPU-hour ceiling.
    2. `BUDGET_TOTAL_GPU_HOURS` = 20 is not greater than
       `BUDGET_HOURLY_GPU_HOURS` = 20, so any usage large enough to trip the
       hourly rule has already tripped the cumulative one, which *refuses*
       instead of degrading.

    So the shipped configuration refuses rather than degrades, and the
    graceful fan-out withdrawal is dormant. That is a safe failure direction,
    but it should be a known one rather than a surprise: to make degradation
    live, raise the total ceiling above the hourly one.
    """
    assert app.BUDGET_HOURLY_GPU_HOURS >= float(app.INFER_MAX_CONTAINERS)
    assert app.BUDGET_TOTAL_GPU_HOURS <= app.BUDGET_HOURLY_GPU_HOURS
