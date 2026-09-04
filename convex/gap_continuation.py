"""The small, explicit signal used by the gap-continuation research profile."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

import pandas as pd

from convex.errors import DataError


@dataclass(frozen=True)
class GapContinuationSignal:
    """A direction known using completed bars through 10:00 New York time."""

    session_date: date
    direction: int
    gap: float
    price_at_signal: float
    vwap: float

    @property
    def side(self) -> str:
        return "up" if self.direction > 0 else "down"

    def as_dict(self) -> dict[str, float | str]:
        return {
            "profile": "gap_continuation_vertical",
            "session_date": self.session_date.isoformat(),
            "direction": self.side,
            "overnight_gap_pct": round(self.gap * 100.0, 4),
            "price_at_signal": round(self.price_at_signal, 4),
            "vwap_at_signal": round(self.vwap, 4),
        }


def signal(
    bars: pd.DataFrame,
    prior_close: float,
    minimum_gap: float = 0.003,
    signal_time: time = time(10, 0),
    minimum_vwap_distance: float = 0.0,
) -> GapContinuationSignal | None:
    """Return the continuation direction, or ``None`` when the rule does not fire.

    ``bars`` must contain completed regular-session minute bars for one session,
    indexed by timestamps in the exchange timezone. Requiring the exact signal
    minute prevents stale data from being promoted into a new entry.
    """
    if prior_close <= 0.0:
        raise DataError(f"prior close must be positive, found {prior_close}")
    if minimum_gap <= 0.0:
        raise DataError(f"minimum gap must be positive, found {minimum_gap}")
    if minimum_vwap_distance < 0.0:
        raise DataError(
            f"minimum VWAP distance must not be negative, found {minimum_vwap_distance}"
        )
    required = {"open", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise DataError(f"minute bars are missing {sorted(missing)}")
    session = bars.between_time("09:30", signal_time.isoformat(timespec="minutes"))
    if session.empty or session.index[-1].time() != signal_time:
        raise DataError(f"no completed {signal_time:%H:%M} signal bar")
    volume = session["volume"].astype(float)
    if (volume < 0.0).any() or float(volume.sum()) <= 0.0:
        raise DataError("signal bars have no usable volume")
    opening = float(session.iloc[0]["open"])
    price = float(session.iloc[-1]["close"])
    vwap = float((session["close"].astype(float) * volume).sum() / volume.sum())
    gap = opening / prior_close - 1.0
    direction = (
        1
        if gap >= minimum_gap and price > vwap * (1.0 + minimum_vwap_distance)
        else -1
        if gap <= -minimum_gap and price < vwap * (1.0 - minimum_vwap_distance)
        else 0
    )
    if direction == 0:
        return None
    return GapContinuationSignal(session.index[-1].date(), direction, gap, price, vwap)
