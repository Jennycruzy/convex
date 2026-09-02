"""Rebuild a history of implied variance from the option tape.

The volatility regime rule asks whether today's implied variance is high or low
against its own recent history. It has never had that history. `ScenarioSet`
supplies realised variance instead, which `annualised_variance` documents
plainly as a substitute made because recorded implied variance "does not exist
until the agent has run for weeks".

The substitute has a direction. Implied variance sits above realised on average,
which is the variance risk premium, so today's implied reading lands in the
upper part of a realised distribution nearly every time. In every reading the
live ledger holds, the rule has returned high_variance or middle and has never
once returned low_variance, which leaves the two families favoured in that
regime, call_bwb and debit_vertical, unable to be selected at all.

The history does not have to be waited for. `reconstruct.features` already
solves implied variance out of the tape for every training row, on the same
bisection the live path uses since the vendor stopped publishing Greeks on
expiration day. This walks that path and writes the series down, so the
comparison can be implied against implied.

Writing the series is all this does. It changes no threshold and no decision.

Run it with:  .venv/bin/python -m scripts.variance_history --days 800
              .venv/bin/python -m scripts.variance_history --days 30 --out /tmp/probe.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

from convex import reconstruct
from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError, DataError
from convex.reconstruct import build as rebuild


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=800, help="calendar days to walk back")
    parser.add_argument("--out", help="where to write the series (default data/variance/implied.json)")
    arguments = parser.parse_args()
    if arguments.days <= 0:
        raise DataError("--days must be positive")

    config = load()
    symbol = config.str_("underlying.symbol")
    zone = ZoneInfo(config.str_("session.timezone"))
    gateway = AlpacaGateway(config)

    entry_time = dtime.fromisoformat(config.str_("reconstruction.entry_time"))
    close_time = dtime.fromisoformat(config.str_("session.close_time"))
    minimum = config.float_("reconstruction.min_coverage")

    now, _ = gateway.clock()
    end = reconstruct.last_rebuildable_session(now, close_time, zone)
    start = end - timedelta(days=arguments.days)
    sessions = gateway.sessions(start, end)
    if not sessions:
        raise DataError(f"Alpaca's calendar lists no session between {start} and {end}")
    print(f"{len(sessions)} sessions on the calendar from {start} to {end}")

    bars = gateway.minute_bars(
        symbol,
        datetime.combine(start, dtime(0, 0), tzinfo=zone),
        datetime.combine(end, dtime(23, 59), tzinfo=zone),
    )
    print(f"{len(bars):,} underlying minute bars\n")

    series: list[dict] = []
    thin = unusable = 0
    for session in sessions:
        day = session.session_date
        entry_at = datetime.combine(day, entry_time, tzinfo=zone)
        close_at = datetime.combine(day, close_time, tzinfo=zone)
        before_entry = bars[bars.index <= entry_at]
        before_close = bars[bars.index <= close_at]
        if before_entry.empty or before_close.empty:
            continue
        spot_at_entry = float(before_entry.iloc[-1]["close"])
        spot_at_close = float(before_close.iloc[-1]["close"])

        try:
            rebuilt = rebuild(
                gateway, config, day, entry_at, close_at, spot_at_entry, spot_at_close
            )
        except ConvexError as error:
            if unusable == 0:
                print(f"  first session that would not rebuild, {day}: {error}")
            unusable += 1
            continue
        if rebuilt.coverage < minimum:
            thin += 1
            continue
        try:
            # The same computation the training rows use, on the same code path,
            # so a reading here and a reading in a fitted row cannot disagree.
            variance_up, variance_dn = reconstruct.implied_variance(
                rebuilt, entry_at, close_at
            )
        except ConvexError as error:
            # Named, not counted. A silent tally of failures is how a rebuild
            # that never worked looks exactly like a market with no data in it.
            if unusable == 0:
                print(f"  first unusable session, {day}: {error}")
            unusable += 1
            continue
        series.append(
            {
                "session": day.isoformat(),
                "iv_total": variance_up + variance_dn,
                "iv_up": variance_up,
                "iv_dn": variance_dn,
                "spot_at_entry": spot_at_entry,
                "coverage": rebuilt.coverage,
            }
        )

    if not series:
        raise DataError("not one session in the window could be rebuilt")

    readings = [row["iv_total"] for row in series]
    print(f"rebuilt {len(series)} sessions with implied variance")
    print(f"  {thin} dropped for thin ladder coverage, {unusable} unusable")
    print()
    print(f"  min    {min(readings):.6f}")
    print(f"  p40    {statistics.quantiles(readings, n=100)[39]:.6f}")
    print(f"  median {statistics.median(readings):.6f}")
    print(f"  p60    {statistics.quantiles(readings, n=100)[59]:.6f}")
    print(f"  max    {max(readings):.6f}")

    destination = Path(arguments.out) if arguments.out else (
        config.path_("paths.chain_archive").parent / "variance" / "implied.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "built_at": datetime.now(tz=zone).isoformat(),
                "symbol": symbol,
                "entry_time": entry_time.isoformat(),
                "close_time": close_time.isoformat(),
                "sessions": len(series),
                "note": (
                    "Implied variance at the entry minute, solved from the option tape by "
                    "the same code the training rows use. Not recorded live: these sessions "
                    "were rebuilt, so the book is gone and only prints survive."
                ),
                "series": series,
            },
            indent=2,
        )
    )
    print(f"\nwritten to {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
