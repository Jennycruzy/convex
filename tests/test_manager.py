"""What the guard does, and what it refuses to do.

The assignment guard is the one piece of this project with no counterpart in
the research it implements: SPXW settles in cash, SPY settles in shares. A
guard that has never been observed firing does not exist, so every branch of it
is exercised here against literal positions with known answers, and against the
paper account in tests/integration.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from convex.errors import DataError, ExecutionError
from convex.instruments import OptionContract, Quote, Right, parse_occ_symbol
from convex.manager import (
    OpenLeg,
    Trigger,
    closing_limit,
    legs_to_close,
    read_positions,
    settlement_pnl,
)


def open_leg(strike: float, right: Right, contracts: int, entry_price: float = 1.0) -> OpenLeg:
    tag = "P" if right is Right.PUT else "C"
    symbol = f"SPY260828{tag}{int(strike * 1000):08d}"
    return OpenLeg(
        contract=parse_occ_symbol(symbol, 100),
        contracts=contracts,
        average_entry_price=entry_price,
        unrealised_pnl=0.0,
    )


def quote(symbol: str, bid: float, ask: float) -> Quote:
    return Quote(symbol, bid, ask, 25, 25, datetime.now(timezone.utc))


class FakePosition:
    """A position record shaped exactly like the ones Alpaca returns."""

    def __init__(self, symbol, qty, asset_class="us_option", avg=1.0, pl=0.0):
        self.symbol = symbol
        self.qty = str(qty)
        self.asset_class = asset_class
        self.avg_entry_price = str(avg)
        self.unrealized_pl = str(pl)


# ------------------------------------------------------------------ occ symbols


def test_occ_symbol_round_trips_strike_right_and_expiry():
    contract = parse_occ_symbol("SPY260828P00647500", 100)
    assert contract.right is Right.PUT
    assert contract.strike == 647.5
    assert contract.expiry == date(2026, 8, 28)
    assert contract.multiplier == 100


@pytest.mark.parametrize(
    "symbol",
    ["SPY260828X00650000", "SPY26082P00650000", "260828P00650000", "SPYP00650000"],
)
def test_a_symbol_it_cannot_parse_raises_rather_than_guessing(symbol):
    with pytest.raises(DataError):
        parse_occ_symbol(symbol, 100)


# -------------------------------------------------------------------- positions


def test_reading_positions_keeps_options_and_signs_the_quantity():
    legs = read_positions(
        [
            FakePosition("SPY260828P00650000", 1),
            FakePosition("SPY260828P00645000", -2),
        ],
        "SPY",
        100,
    )
    assert [leg.contracts for leg in legs] == [1, -2]
    assert legs[1].is_short


def test_an_assigned_share_position_stops_the_manager_rather_than_being_skipped():
    with pytest.raises(ExecutionError, match="assignment has already occurred"):
        read_positions(
            [FakePosition("SPY", 100, asset_class="us_equity")], "SPY", 100
        )


# ------------------------------------------------------------------- the guard


def test_nothing_closes_while_no_trigger_has_fired():
    legs = [open_leg(650, Right.PUT, 1), open_leg(645, Right.PUT, -2)]
    assert legs_to_close(legs, [], spot=649.0, pin_band=1.0) == []


def test_the_guard_closes_only_what_can_settle_into_shares():
    itm_short = open_leg(645.0, Right.PUT, -2)
    far_otm_long = open_leg(600.0, Right.PUT, 1)
    selected = legs_to_close(
        [far_otm_long, itm_short], [Trigger.ASSIGNMENT_GUARD], spot=640.0, pin_band=0.96
    )
    assert selected == [itm_short]


def test_a_leg_pinned_to_its_strike_counts_as_at_risk_even_when_out_of_the_money():
    pinned = open_leg(650.0, Right.PUT, -1)
    selected = legs_to_close(
        [pinned], [Trigger.ASSIGNMENT_GUARD], spot=650.60, pin_band=650.60 * 0.0015
    )
    assert selected == [pinned]


def test_the_kill_switch_takes_everything_including_worthless_longs():
    legs = [open_leg(600.0, Right.PUT, 1), open_leg(645.0, Right.PUT, -2)]
    assert len(legs_to_close(legs, [Trigger.KILL_SWITCH], spot=650.0, pin_band=1.0)) == 2


def test_shorts_are_always_closed_before_longs():
    legs = [
        open_leg(650.0, Right.PUT, 1),
        open_leg(645.0, Right.PUT, -2),
        open_leg(635.0, Right.PUT, 1),
    ]
    order = legs_to_close(legs, [Trigger.DAILY_LOSS_LIMIT], spot=640.0, pin_band=1.0)
    assert order[0].is_short
    assert not any(leg.is_short for leg in order[1:])


# ------------------------------------------------------------------ close price


def test_a_long_is_sold_into_the_bid_and_a_short_is_bought_from_the_ask():
    long_leg = open_leg(650.0, Right.PUT, 1)
    short_leg = open_leg(645.0, Right.PUT, -2)
    book = quote("x", 2.00, 2.10)
    assert closing_limit(long_leg, book) == 2.00
    assert closing_limit(short_leg, book) == 2.10


def test_closing_a_long_with_no_bid_raises_rather_than_pricing_it_at_zero():
    with pytest.raises(DataError, match="no bid to close against"):
        closing_limit(open_leg(600.0, Right.PUT, 1), quote("x", 0.0, 0.05))


# ------------------------------------------------------------------- settlement


def test_a_broken_wing_butterfly_entered_for_a_credit_keeps_it_above_every_strike():
    legs = [
        (parse_occ_symbol("SPY260828P00650000", 100), 1),
        (parse_occ_symbol("SPY260828P00645000", 100), -2),
        (parse_occ_symbol("SPY260828P00635000", 100), 1),
    ]
    # Entered for a 0.20 credit, so the debit is negative.
    assert settlement_pnl(legs, -0.20, 1, 660.0, 100) == pytest.approx(20.0)


def test_the_same_structure_bottoms_out_at_its_computed_worst_case():
    legs = [
        (parse_occ_symbol("SPY260828P00650000", 100), 1),
        (parse_occ_symbol("SPY260828P00645000", 100), -2),
        (parse_occ_symbol("SPY260828P00635000", 100), 1),
    ]
    # Below the lowest strike the wings are 1x1 and the loss is the wing gap of
    # 5 points against the 0.20 credit, per share, on a hundred-share contract.
    assert settlement_pnl(legs, -0.20, 1, 600.0, 100) == pytest.approx(-480.0)


def test_settling_a_structure_of_no_contracts_raises():
    legs = [(parse_occ_symbol("SPY260828P00650000", 100), 1)]
    with pytest.raises(DataError):
        settlement_pnl(legs, 1.0, 0, 650.0, 100)
