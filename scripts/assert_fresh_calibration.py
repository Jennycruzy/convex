"""Fail unless today's live calibration receipt is fresh enough for the entry.

The session timer runs at exactly 10:00 ET. Calibration runs five minutes
earlier under its own timer; this assertion prevents the decision service from
using a stale, closed-market, or failed calibration if that earlier job did not
produce a valid receipt.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError, DataError
from convex.ledger import Action, Ledger, Record, new_cycle_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-age-seconds", type=float, default=900.0)
    arguments = parser.parse_args()
    if arguments.max_age_seconds <= 0:
        raise DataError("max-age-seconds must be positive")

    config = load()
    ledger = Ledger(config.path_("paths.ledger"))
    try:
        gateway = AlpacaGateway(config)
        now, market_open = gateway.clock()
        if not market_open:
            raise DataError("Alpaca reports the market closed; no entry may run")

        latest = None
        for record in ledger.read():
            if record.get("action") == Action.CALIBRATION.value:
                latest = record
        if latest is None:
            raise DataError("no calibration receipt exists for today's entry")

        outcome = latest.get("outcome") or {}
        try:
            stamped = datetime.fromisoformat(str(latest["ts"]).replace("Z", "+00:00"))
        except (KeyError, ValueError) as error:
            raise DataError("latest calibration receipt has no valid timestamp") from error
        age = (datetime.now(timezone.utc) - stamped.astimezone(timezone.utc)).total_seconds()
        if stamped.astimezone(now.tzinfo).date() != now.date():
            raise DataError("latest calibration receipt is not from today's exchange session")
        if outcome.get("key") != "liquidity.max_relative_spread":
            raise DataError(
                "latest calibration receipt did not successfully write the liquidity threshold"
            )
        if not bool(outcome.get("market_open")):
            raise DataError("latest calibration receipt was not taken while Alpaca reported market open")
        if age > arguments.max_age_seconds:
            raise DataError(
                f"latest calibration is {age:.0f}s old, exceeding {arguments.max_age_seconds:.0f}s"
            )
    except ConvexError as error:
        ledger.append(
            Record(
                action=Action.RISK_HALT,
                cycle_id=new_cycle_id(),
                rationale=f"No 10:00 entry: fresh calibration prerequisite failed: {error}.",
                reject_reason="fresh_calibration",
            )
        )
        raise
    print(f"fresh live calibration receipt {age:.0f}s old; entry may proceed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
