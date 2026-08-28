"""Fixtures that build chain rows out of explicit numbers.

These are not mocks of the Alpaca API. Nothing here stands in for a network
call or invents a market: they are literal contract records and literal prices
used to check that arithmetic with a known closed-form answer is correct. Every
test that needs live market behaviour lives in tests/integration and talks to
the paper account.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from convex.instruments import ChainEntry, Greeks, Leg, OptionContract, Quote, Right

EXPIRY = date(2026, 8, 28)


def contract(strike: float, right: Right, symbol: str | None = None) -> OptionContract:
    tag = "P" if right is Right.PUT else "C"
    return OptionContract(
        symbol=symbol or f"SPY260828{tag}{int(strike * 1000):08d}",
        underlying="SPY",
        right=right,
        strike=strike,
        expiry=EXPIRY,
        multiplier=100,
    )


def entry(
    strike: float,
    right: Right,
    bid: float,
    ask: float,
    *,
    size: int = 25,
    open_interest: int = 5_000,
    greeks: Greeks | None = None,
    age_seconds: float = 0.0,
) -> ChainEntry:
    timestamp = datetime.now(timezone.utc).timestamp() - age_seconds
    return ChainEntry(
        contract=contract(strike, right),
        quote=Quote(
            symbol=contract(strike, right).symbol,
            bid=bid,
            ask=ask,
            bid_size=size,
            ask_size=size,
            timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc),
        ),
        greeks=greeks,
        open_interest=open_interest,
        volume=1_000,
    )


def leg(strike: float, right: Right, bid: float, ask: float, ratio: int, **kwargs) -> Leg:
    return Leg(entry=entry(strike, right, bid, ask, **kwargs), ratio=ratio)


@pytest.fixture
def put_bwb_legs() -> list[Leg]:
    """A put broken-wing butterfly: +1 x 650P, -2 x 645P, +1 x 635P.

    The far wing is wider than the near wing, which is what makes the payoff
    flat and positive above the highest strike when the structure is entered
    for a credit, and what bounds the loss below the lowest strike.
    """
    return [
        leg(650.0, Right.PUT, 3.00, 3.10, +1),
        leg(645.0, Right.PUT, 1.60, 1.70, -2),
        leg(635.0, Right.PUT, 0.40, 0.48, +1),
    ]
