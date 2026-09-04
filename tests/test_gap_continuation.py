from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from convex.agent import CycleResult
from convex.config import load
from convex.errors import DataError
from convex.gap_continuation import signal

ZONE = ZoneInfo("America/New_York")


def bars(opens, closes, volumes):
    index = pd.date_range(datetime(2026, 9, 3, 9, 30, tzinfo=ZONE), periods=len(opens), freq="min")
    return pd.DataFrame({"open": opens, "close": closes, "volume": volumes}, index=index)


def test_up_gap_above_vwap_signals_up():
    frame = bars([103.0] * 31, [103.0] * 30 + [104.0], [10] * 31)
    found = signal(frame, prior_close=100.0)
    assert found is not None
    assert found.side == "up"
    assert found.gap == pytest.approx(0.03)


def test_down_gap_below_vwap_signals_down():
    frame = bars([97.0] * 31, [97.0] * 30 + [96.0], [10] * 31)
    found = signal(frame, prior_close=100.0)
    assert found is not None
    assert found.side == "down"


def test_gap_that_loses_vwap_stands_down():
    frame = bars([103.0] * 31, [103.0] * 30 + [102.0], [10] * 31)
    assert signal(frame, prior_close=100.0) is None


def test_missing_completed_signal_bar_refuses():
    frame = bars([103.0] * 30, [103.0] * 30, [10] * 30)
    with pytest.raises(DataError, match="completed 10:00"):
        signal(frame, prior_close=100.0)


def test_active_profile_passes_the_configured_reprice_ladder(monkeypatch):
    """The scheduled profile must not bypass the retry policy in config."""
    import scripts.run_gap_continuation as runner

    class Gateway:
        def __init__(self, config):
            self.config = config

        def clock(self):
            return datetime(2026, 9, 3, 10, 2, tzinfo=ZONE), True

    class Ledger:
        def __init__(self, path):
            self.path = path

        def read(self):
            return []

    class Returns:
        def tolist(self):
            return []

    class Scenarios:
        log_returns = Returns()

    class Found:
        direction = 1

        def as_dict(self):
            return {"direction": "up"}

    class CapturingAgent:
        kwargs = None

        def __init__(self, **kwargs):
            CapturingAgent.kwargs = kwargs

        def run_cycle(self, **kwargs):
            return CycleResult("test-cycle", True, "test")

    monkeypatch.setattr(runner, "AlpacaGateway", Gateway)
    monkeypatch.setattr(runner, "Ledger", Ledger)
    monkeypatch.setattr(runner, "observed_signal", lambda *args, **kwargs: Found())
    monkeypatch.setattr(runner, "build_scenarios", lambda *args, **kwargs: Scenarios())
    monkeypatch.setattr(runner, "save_scenarios", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "Agent", CapturingAgent)
    monkeypatch.setattr(runner.sys, "argv", ["run_gap_continuation"])

    assert runner.main() == 0
    assert CapturingAgent.kwargs["reprice_ticks"] == tuple(
        int(tick) for tick in load().list_("execution.reprice_ticks")
    )
    assert CapturingAgent.kwargs["decision_probability"] == load().float_(
        "strategy.gap_continuation.admission_score"
    )
