from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from convex.errors import DataError
from convex.gap_continuation import signal

ZONE = ZoneInfo("America/New_York")


def bars(opens, closes, volumes):
    index = pd.date_range(
        datetime(2026, 9, 3, 9, 30, tzinfo=ZONE), periods=len(opens), freq="min"
    )
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
