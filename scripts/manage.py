"""Watch the open positions through the session and settle them after it.

This runs alongside scripts.run_cycle, not instead of it. The cycle opens
positions at 10:00 and never touches them again; this is what stands between an
open 0DTE structure and a hundred shares of SPY arriving in the account
overnight, and what writes each structure's result back into the ledger so the
next day's features have something to read.

Run it with:  .venv/bin/python -m scripts.manage
              .venv/bin/python -m scripts.manage --settle    (after the close)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta

from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError
from convex.ledger import Ledger
from convex.manager import PositionManager


def official_close(gateway: AlpacaGateway, symbol: str, session) -> float:
    """The last traded price of the session, from the market data API.

    Settlement is arithmetic with one input and this is that input, so it is
    read rather than remembered: the agent's own last quote from 15:59 is not
    where the session closed.
    """
    bars = gateway.minute_bars(symbol, session.open_at, session.close_at + timedelta(minutes=1))
    if not bars:
        raise ConvexError(
            f"no {symbol} bars for {session.session_date}; the session cannot be settled"
        )
    return float(bars[-1].close)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settle",
        action="store_true",
        help="record each structure's expiry result instead of watching",
    )
    parser.add_argument(
        "--once", action="store_true", help="make a single pass rather than polling"
    )
    arguments = parser.parse_args()

    config = load()
    gateway = AlpacaGateway(config)
    ledger = Ledger(config.path_("paths.ledger"))
    manager = PositionManager(gateway, config, ledger)
    symbol = config.str_("underlying.symbol")

    now, market_open = gateway.clock()
    sessions = gateway.sessions(now.date(), now.date())
    if not sessions:
        print(f"no session on {now.date()}; nothing to watch or settle")
        return 0
    session = sessions[0]

    if arguments.settle:
        price = official_close(gateway, symbol, session)
        print(f"{symbol} settled at {price:.2f} on {session.session_date}")
        results = manager.settle(session.session_date, price)
        if not results:
            print("nothing opened today is waiting to be settled")
        for result in results:
            if "unsettled" in result:
                print(f"  {result['structure']:<16} unsettled: {result['unsettled']}")
            else:
                print(f"  {result['structure']:<16} {result['realised_pnl']:+,.2f}")
        return 0

    poll = config.float_("session.manager_poll_seconds")
    while True:
        now, market_open = gateway.clock()
        if not market_open:
            print(f"{now:%H:%M:%S} market closed; stopping")
            return 0
        report = manager.review(now, session.close_at)
        minutes = (session.close_at - now).total_seconds() / 60.0
        print(f"{now:%H:%M:%S}  {minutes:5.1f}m to close  {report.reason}")
        for closed in report.closed:
            print(f"    closed {closed['symbol']} as {closed['order_id']}")
        for failure in report.failed:
            print(f"    COULD NOT CLOSE {failure['symbol']}: {failure['error']}")
        if arguments.once:
            return 1 if report.failed else 0
        if report.failed:
            # A leg the guard wanted gone is still open. Nothing here can fix
            # that by trying harder, and looping quietly on it would hide it.
            return 1
        time.sleep(poll)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
