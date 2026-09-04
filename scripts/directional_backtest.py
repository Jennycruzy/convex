"""Read-only walk-forward test of low-cost SPY intraday directional rules.

This program never creates a gateway order object, writes the ledger, or changes
configuration.  It uses completed one-minute stock bars only: an entry signal
at 10:00 ET and an exit at 15:55 ET.  Candidate rules differ only in direction
(continuation or reversal) and two observable thresholds.  Selection occurs on
the earlier segment; the final segment is untouched until selection is complete.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError, DataError

ENTRY = time(10, 0)
EXIT = time(15, 55)


@dataclass(frozen=True, order=True)
class Rule:
    mode: str
    gap: float
    vwap_distance: float

    def label(self) -> str:
        return f"{self.mode:12} gap>={self.gap:.3%} vwap>={self.vwap_distance:.3%}"


@dataclass(frozen=True)
class Observation:
    day: date
    gap: float
    vwap_distance: float
    forward_return: float


def _day(frame: pd.DataFrame, day: date) -> pd.DataFrame:
    return frame[
        (frame.index.date == day) & (frame.index.time >= time(9, 30)) & (frame.index.time <= EXIT)
    ]


def observations(bars: pd.DataFrame) -> list[Observation]:
    """Derive opportunities without reading any bar after the entry signal."""
    if not {"open", "close", "volume"} <= set(bars.columns):
        raise DataError("stock bars must include open, close, and volume")
    days = sorted(set(bars.index.date))
    rows: list[Observation] = []
    previous: pd.DataFrame | None = None
    for day in days:
        current = _day(bars, day)
        if previous is not None and not current.empty:
            signal_bars = current[current.index.time <= ENTRY]
            exit_bars = current[current.index.time <= EXIT]
            if (
                not signal_bars.empty
                and signal_bars.index[-1].time() == ENTRY
                and not exit_bars.empty
                and exit_bars.index[-1].time() == EXIT
            ):
                volume = signal_bars["volume"].astype(float)
                if float(volume.sum()) > 0.0:
                    vwap = float((signal_bars["close"].astype(float) * volume).sum() / volume.sum())
                    price = float(signal_bars.iloc[-1]["close"])
                    rows.append(
                        Observation(
                            day,
                            float(current.iloc[0]["open"]) / float(previous.iloc[-1]["close"])
                            - 1.0,
                            price / vwap - 1.0,
                            float(exit_bars.iloc[-1]["close"]) / price - 1.0,
                        )
                    )
        if not current.empty and current.index[-1].time() == EXIT:
            previous = current
    return rows


def returns(rows: list[Observation], rule: Rule, cost_bps: float) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        aligned = row.gap * row.vwap_distance > 0.0
        if abs(row.gap) < rule.gap or abs(row.vwap_distance) < rule.vwap_distance or not aligned:
            continue
        direction = 1.0 if row.gap > 0.0 else -1.0
        if rule.mode == "reversal":
            direction *= -1.0
        values.append(direction * row.forward_return - cost_bps / 10_000.0)
    return np.asarray(values, dtype=float)


def summary(values: np.ndarray) -> tuple[int, float, float | None]:
    if not len(values):
        return 0, 0.0, None
    total = float(values.sum())
    if len(values) < 2:
        return len(values), total, None
    lower = float(
        values.mean() - NormalDist().inv_cdf(0.95) * values.std(ddof=1) / math.sqrt(len(values))
    )
    return len(values), total, lower


def historical_bars(
    gateway: AlpacaGateway, symbol: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """Fetch bounded windows because the MCP transport drops oversized responses."""
    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=90), end)
        frames.append(gateway.minute_bars(symbol, cursor, chunk_end))
        cursor = chunk_end + timedelta(minutes=1)
    combined = pd.concat(frames).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=400, help="calendar days of bars to request")
    parser.add_argument(
        "--holdout", type=int, default=60, help="final sessions reserved from selection"
    )
    parser.add_argument(
        "--cost-bps", type=float, default=2.0, help="round-trip cost deducted per trade"
    )
    arguments = parser.parse_args()
    if arguments.days <= arguments.holdout or arguments.holdout < 20 or arguments.cost_bps < 0.0:
        raise DataError("days must exceed holdout (at least 20), and cost-bps must be non-negative")
    config = load()
    zone = ZoneInfo(config.str_("session.timezone"))
    end = datetime.now(zone).replace(hour=23, minute=59, second=0, microsecond=0)
    start = end - timedelta(days=arguments.days)
    print("research only: no orders, ledger writes, or config changes")
    with AlpacaGateway(config) as gateway:
        bars = historical_bars(gateway, config.str_("underlying.symbol"), start, end).tz_convert(
            zone
        )
    rows = observations(bars)
    if len(rows) <= arguments.holdout:
        raise DataError(f"only {len(rows)} complete sessions; need more than holdout")
    train, holdout = rows[: -arguments.holdout], rows[-arguments.holdout :]
    rules = [
        Rule(mode, gap, vwap)
        for mode in ("continuation", "reversal")
        for gap in (0.001, 0.002, 0.003, 0.005)
        for vwap in (0.0, 0.0005, 0.001)
    ]
    scored = [(rule, *summary(returns(train, rule, arguments.cost_bps))) for rule in rules]
    eligible = [row for row in scored if row[1] >= 20 and row[3] is not None]
    if not eligible:
        raise DataError("no rule generated 20 training trades")
    chosen = max(eligible, key=lambda row: (row[3], row[2]))
    rule, train_n, train_total, train_lb = chosen
    test_n, test_total, test_lb = summary(returns(holdout, rule, arguments.cost_bps))
    print(
        f"sessions={len(rows)} train={len(train)} holdout={len(holdout)} cost={arguments.cost_bps:.2f} bps"
    )
    print(f"chosen from training only: {rule.label()}")
    print(f"train: n={train_n} net return={train_total:.3%} 95% lower mean={train_lb:.3%}")
    print(
        f"holdout: n={test_n} net return={test_total:.3%} 95% lower mean={'n/a' if test_lb is None else f'{test_lb:.3%}'}"
    )
    promote = test_n >= 15 and test_total > 0.0 and test_lb is not None and test_lb > 0.0
    print("PROMOTE FOR EXECUTION DESIGN" if promote else "DO NOT PROMOTE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
