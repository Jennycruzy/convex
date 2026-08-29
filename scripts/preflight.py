"""Day-one verification against the live paper account.

Nothing in this project is allowed to assume a market fact, so before anything
trades, every fact it depends on is fetched and printed: the account and its
equity, Alpaca's own clock and calendar across the competition window, which
SPY expiries actually exist, and whether the chain comes back with the Greeks,
implied volatility and open interest the feature engine needs.

Run it with:  .venv/bin/python -m scripts.preflight
It exits non-zero if any of that is missing, and writes what it found to the
ledger so the receipt for day one is the same artefact as every other receipt.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError
from convex.ledger import Action, Ledger, Record, new_cycle_id

EXPECTED_STARTING_EQUITY = 100_000.0
COMPETITION_START = date(2026, 8, 28)
COMPETITION_END = date(2026, 9, 4)


def main() -> int:
    config = load()
    symbol = config.str_("underlying.symbol")
    gateway = AlpacaGateway(config)
    ledger = Ledger(config.path_("paths.ledger"))
    cycle_id = new_cycle_id()
    findings: dict[str, object] = {}
    problems: list[str] = []

    account = gateway.account()
    findings["account"] = {
        "account_number": account.account_number,
        "status": account.status,
        "equity": account.equity,
        "cash": account.cash,
        "buying_power": account.buying_power,
        "options_buying_power": account.options_buying_power,
        "options_approved_level": account.options_approved_level,
        "pattern_day_trader": account.pattern_day_trader,
    }
    print(f"account          {account.account_number}  status {account.status}")
    print(f"equity           {account.equity:,.2f}   cash {account.cash:,.2f}")
    print(f"buying power     {account.buying_power:,.2f}   options {account.options_buying_power:,.2f}")
    print(f"options level    {account.options_approved_level}")

    if abs(account.equity - EXPECTED_STARTING_EQUITY) > 0.005:
        problems.append(
            f"equity is {account.equity:,.2f}, not the {EXPECTED_STARTING_EQUITY:,.2f} the "
            "submission rules require of a fresh paper account"
        )
    if account.options_approved_level < 3:
        problems.append(
            f"options approval is level {account.options_approved_level}; multi-leg "
            "defined-risk spreads need level 3"
        )

    now, is_open = gateway.clock()
    print(f"\nexchange clock   {now:%Y-%m-%d %H:%M:%S %Z}   market {'open' if is_open else 'closed'}")

    sessions = gateway.sessions(COMPETITION_START, COMPETITION_END)
    open_days = [session.session_date for session in sessions]
    findings["sessions"] = [day.isoformat() for day in open_days]
    print("sessions in the competition window:")
    for session in sessions:
        print(f"  {session.session_date}  {session.open_at:%H:%M} to {session.close_at:%H:%M}")
    for day in (COMPETITION_START + timedelta(days=offset) for offset in range(8)):
        if day <= COMPETITION_END and day.weekday() < 5 and day not in open_days:
            print(f"  {day} is a weekday with no session")

    expiries = gateway.expirations(symbol, now.date())
    findings["expiries"] = [day.isoformat() for day in expiries[:8]]
    print(f"\n{symbol} expiries listed from today: " + ", ".join(str(day) for day in expiries[:8]))
    if expiries[0] != now.date():
        print(f"  note: there is no same-day expiry today; the nearest is {expiries[0]}")

    spot, quoted_at = gateway.spot(symbol)
    print(f"{symbol} spot        {spot:,.2f}  quoted {quoted_at:%H:%M:%S %Z}")

    low = spot * config.float_("candidates.moneyness_low")
    high = spot * config.float_("candidates.moneyness_high")
    wing = spot * config.float_("candidates.max_wing_width_pct")
    chain = gateway.chain(symbol, expiries[0], low - wing, high + wing)

    with_greeks = [row for row in chain if row.greeks is not None]
    with_open_interest = [row for row in chain if row.open_interest is not None]
    relative = sorted(row.quote.relative_spread for row in chain)
    findings["chain"] = {
        "expiry": expiries[0].isoformat(),
        "contracts": len(chain),
        "with_greeks": len(with_greeks),
        "with_open_interest": len(with_open_interest),
        "median_relative_spread": relative[len(relative) // 2],
        "widest_relative_spread": relative[-1],
        "strike_low": chain[0].contract.strike,
        "strike_high": chain[-1].contract.strike,
        "multiplier": chain[0].contract.multiplier,
    }
    print(
        f"\nchain            {len(chain)} contracts on {expiries[0]}, "
        f"{len(with_greeks)} with Greeks, {len(with_open_interest)} with open interest"
    )
    print(
        f"relative spread  median {relative[len(relative) // 2]:.1%}, "
        f"widest {relative[-1]:.1%}"
    )
    print(f"multiplier       {chain[0].contract.multiplier} (measured, not assumed)")

    if not with_greeks:
        problems.append(
            f"the {config.str_('data.options_feed')} options feed returned no Greeks; "
            "the feature engine cannot run without them"
        )
    if not with_open_interest:
        problems.append("no open interest came back, so the exposure features cannot be built")

    ledger.append(
        Record(
            action=Action.CALIBRATION,
            cycle_id=cycle_id,
            rationale=(
                f"Preflight against the live paper account: {symbol} at {spot:,.2f}, "
                f"{len(chain)} contracts on the {expiries[0]} expiry, "
                f"{len(with_greeks)} of them with Greeks."
            ),
            outcome=findings,
            reject_reason="; ".join(problems) if problems else None,
        )
    )

    if problems:
        print("\nproblems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\npreflight passed: every market fact this project depends on came back live.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
