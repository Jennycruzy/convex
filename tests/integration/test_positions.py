"""What the account is holding, and what the guard would do about it.

The assignment guard is the one component with no counterpart in the research
being implemented: SPXW settles in cash, SPY settles in shares. It is also the
component that, if wrong, produces a hundred shares of stock per contract
overnight. So it is exercised against whatever the account really holds, and
against a share position it must refuse to trade around.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from convex.errors import ExecutionError
from convex.manager import PositionManager, Trigger, legs_to_close, read_positions
from tests.integration.conftest import needs_account


@needs_account
def test_positions_parse_into_legs_the_guard_can_reason_about(gateway, config):
    raw = gateway.positions()
    symbol = config.str_("underlying.symbol")
    multiplier = config.int_("underlying.contract_multiplier")

    if not raw:
        print("  the account holds nothing right now")
        assert read_positions(raw, symbol, multiplier) == []
        return

    legs = read_positions(raw, symbol, multiplier)
    for held in legs:
        print(f"  {held.symbol}  {held.contracts:+d} @ {held.average_entry_price} "
              f"({held.contract.right} {held.contract.strike:g} exp {held.contract.expiry})")
        assert held.contracts != 0
        assert held.contract.underlying == symbol


@needs_account
def test_the_guard_selects_what_would_settle_into_shares(gateway, config, spot):
    """Run the real selection against the real book, whatever it holds."""
    symbol = config.str_("underlying.symbol")
    multiplier = config.int_("underlying.contract_multiplier")
    legs = read_positions(gateway.positions(), symbol, multiplier)
    if not legs:
        pytest.skip("no open option positions to run the guard against")

    band = spot * config.float_("session.pin_band_pct")
    at_risk = legs_to_close(legs, [Trigger.ASSIGNMENT_GUARD], spot, band)
    print(f"  {symbol} at {spot:.2f}, pin band {band:.2f}")
    print(f"  {len(at_risk)} of {len(legs)} legs would be closed by the guard")
    for held in at_risk:
        print(f"    {held.symbol} strike {held.contract.strike:g}")

    # Shorts must always come first: closing one can only narrow the worst
    # case, and no ordering may leave a short standing without its wing.
    shorts = [index for index, held in enumerate(at_risk) if held.is_short]
    longs = [index for index, held in enumerate(at_risk) if not held.is_short]
    assert not shorts or not longs or max(shorts) < min(longs)


@needs_account
def test_a_close_can_be_priced_against_the_live_book(gateway, config):
    """The guard needs a two-sided quote for every leg it wants gone."""
    from convex.manager import closing_limit

    symbol = config.str_("underlying.symbol")
    legs = read_positions(gateway.positions(), symbol, config.int_("underlying.contract_multiplier"))
    if not legs:
        pytest.skip("no open option positions to price a close for")

    quotes = gateway.option_quotes([held.symbol for held in legs])
    for held in legs:
        limit = closing_limit(held, quotes[held.symbol])
        print(f"  {held.symbol}: would close {held.contracts:+d} at {limit:.2f} "
              f"(bid {quotes[held.symbol].bid:.2f} ask {quotes[held.symbol].ask:.2f})")
        assert limit > 0.0


@needs_account
def test_an_assigned_share_position_stops_the_manager(config):
    """Shares mean an assignment already happened; no automated close is safe."""
    class SharePosition:
        symbol = config.str_("underlying.symbol")
        qty = "100"
        asset_class = "us_equity"
        avg_entry_price = "650.0"
        unrealized_pl = "0.0"

    with pytest.raises(ExecutionError, match="assignment has already occurred"):
        read_positions([SharePosition()], config.str_("underlying.symbol"), 100)


@needs_account
def test_the_manager_reviews_without_closing_anything_when_nothing_triggers(
    gateway, config, tmp_path
):
    """A review with no trigger must be a read, never a write."""
    from convex.ledger import Ledger

    ledger = Ledger(tmp_path / "review.jsonl")
    manager = PositionManager(gateway, config, ledger)
    now, is_open = gateway.clock()
    sessions = gateway.sessions(now.date(), now.date())
    if not sessions:
        pytest.skip(f"no session on {now.date()}")

    close_at = sessions[0].close_at
    guard = config.float_("session.assignment_guard_minutes")
    if (close_at - now).total_seconds() / 60.0 <= guard:
        pytest.skip("inside the guard window; this test must not trigger a real close")

    report = manager.review(now, close_at)
    print(f"  {report.reason}")
    assert report.closed == [], "a review with no trigger closed something"
    assert report.failed == []
