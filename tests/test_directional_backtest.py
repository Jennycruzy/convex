from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from scripts.directional_backtest import Observation, Rule, observations, returns, summary

ZONE = ZoneInfo("America/New_York")


def _bars(day: str, opening: float, closes: list[float]) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            datetime.fromisoformat(f"{day}T09:30:00").replace(tzinfo=ZONE),
            datetime.fromisoformat(f"{day}T10:00:00").replace(tzinfo=ZONE),
            datetime.fromisoformat(f"{day}T15:55:00").replace(tzinfo=ZONE),
        ]
    )
    return pd.DataFrame(
        {"open": [opening, opening, opening], "close": closes, "volume": [10, 10, 10]}, index=index
    )


def test_observations_use_no_bar_after_entry_for_signal():
    frame = pd.concat(
        [_bars("2026-08-31", 100, [100, 100, 100]), _bars("2026-09-01", 101, [101, 102, 104])]
    )
    found = observations(frame)
    assert len(found) == 1
    assert found[0].gap == pytest.approx(0.01)
    assert found[0].vwap_distance > 0.0
    assert found[0].forward_return == 104 / 102 - 1


def test_continuation_and_reversal_have_opposite_returns_after_cost():
    rows = [Observation(date(2026, 9, 1), 0.01, 0.01, 0.02)]
    continuation = returns(rows, Rule("continuation", 0.003, 0.0), 2.0)
    reversal = returns(rows, Rule("reversal", 0.003, 0.0), 2.0)
    assert continuation[0] == 0.0198
    assert reversal[0] == -0.0202
    assert summary(continuation) == (1, 0.0198, None)
