"""Rebuilding a past session, and refusing to invent the parts that are gone.

The arithmetic here is checked against itself, in that a price produced by the
pricer must solve back to the volatility that produced it, and the refusals are
checked explicitly, because the failure mode that matters is not a wrong number
but a confident one where there should be none.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from convex import reconstruct
from convex.config import load
from convex.errors import DataError
from convex.instruments import Right
from convex.instruments import OptionContract
from convex.reconstruct import (
    ReconstructedEntry,
    arbitrage_free,
    black_scholes,
    implied_volatility,
    occ_symbol,
    strike_ladder,
)

YEARS = 6.0 / (365.0 * 24.0)  # 10:00 to the close, as a 0DTE contract sees it
RATE = 0.043


def test_the_occ_symbol_is_built_the_way_the_exchange_writes_it():
    assert occ_symbol("SPY", date(2026, 8, 26), Right.PUT, 770.0) == "SPY260826P00770000"
    assert occ_symbol("SPY", date(2026, 8, 26), Right.CALL, 769.5) == "SPY260826C00769500"


def test_a_strike_the_symbol_cannot_express_raises():
    with pytest.raises(DataError, match="thousandths"):
        occ_symbol("SPY", date(2026, 8, 26), Right.CALL, 770.00005)


@pytest.mark.parametrize("vol", [0.08, 0.15, 0.22, 0.40, 0.85])
@pytest.mark.parametrize("right", [Right.CALL, Right.PUT])
def test_a_price_solves_back_to_the_volatility_that_made_it(vol, right):
    spot, strike = 769.30, 770.0
    price = black_scholes(spot, strike, right, YEARS, RATE, vol)
    solved = implied_volatility(price, spot, strike, right, YEARS, RATE)
    assert solved is not None
    assert solved == pytest.approx(vol, abs=1e-3)


def test_put_call_parity_holds_for_the_pricer():
    spot, strike, vol = 769.30, 765.0, 0.20
    call = black_scholes(spot, strike, Right.CALL, YEARS, RATE, vol)
    put = black_scholes(spot, strike, Right.PUT, YEARS, RATE, vol)
    assert call - put == pytest.approx(spot - strike * math.exp(-RATE * YEARS), abs=1e-6)


def test_a_print_at_parity_has_no_implied_volatility_rather_than_a_floor():
    """The deep-in-the-money case that poisoned the skew feature before.

    A single print at intrinsic carries no volatility information. Returning
    the solver's floor would report 0.01% volatility as though it were measured.
    """
    spot, strike = 766.795, 759.0
    assert implied_volatility(spot - strike, spot, strike, Right.CALL, YEARS, RATE) is None


def test_a_print_below_intrinsic_or_above_the_ceiling_does_not_solve():
    assert implied_volatility(1.0, 769.3, 700.0, Right.CALL, YEARS, RATE) is None
    assert implied_volatility(800.0, 769.3, 770.0, Right.CALL, YEARS, RATE) is None
    assert implied_volatility(0.0, 769.3, 770.0, Right.PUT, YEARS, RATE) is None


def test_an_expired_contract_has_no_time_value_left_to_solve():
    assert implied_volatility(5.0, 769.3, 770.0, Right.PUT, 0.0, RATE) is None
    assert black_scholes(769.3, 765.0, Right.CALL, 0.0, RATE, 0.2) == pytest.approx(4.3)


def test_the_ladder_covers_the_configured_band_in_whole_dollars():
    config = load()
    ladder = strike_ladder(769.30, config)
    low = 769.30 * config.float_("candidates.moneyness_low")
    high = 769.30 * config.float_("candidates.moneyness_high")
    assert ladder[0] >= low and ladder[-1] <= high
    assert all(float(s).is_integer() for s in ladder)
    assert ladder == sorted(ladder)
    # A 2% band on a 769 underlying is roughly thirty-one dollars wide.
    assert 25 <= len(ladder) <= 40


# ------------------------------------------------------------ tape features


def test_a_contract_with_no_daily_bar_has_not_traded_rather_than_no_data():
    """The one place absence really is an observation, written down once."""
    from convex.features import traded_volume

    assert traded_volume(None) == 0.0
    assert traded_volume(0) == 0.0
    assert traded_volume(412) == 412.0


def test_the_tape_reports_flow_the_book_cannot_be_asked_for():
    from convex.features import tape_features

    rows = [(Right.PUT, 300), (Right.PUT, 100), (Right.CALL, 100), (Right.CALL, None)]
    tape = tape_features(rows)
    assert tape["tape_put_share"] == pytest.approx(0.8)
    assert tape["tape_breadth"] == pytest.approx(0.75)
    # Herfindahl: 0.6^2 + 0.2^2 + 0.2^2
    assert tape["tape_concentration"] == pytest.approx(0.44)


def test_a_chain_that_did_not_trade_asserts_no_imbalance_rather_than_inventing_one():
    from convex.features import tape_features

    tape = tape_features([(Right.PUT, 0), (Right.CALL, None)])
    assert tape["tape_put_share"] == 0.5
    assert tape["tape_volume"] == 0.0
    assert tape["tape_breadth"] == 0.0


def test_the_tape_features_are_computable_from_both_sides_of_the_seam():
    """A feature meaning one thing in training and another at 10:00 is worse
    than no feature, so the two paths must produce the same names."""
    from convex.features import tape_features
    from convex.reconstruct import RECONSTRUCTED_FEATURES

    produced = set(tape_features([(Right.PUT, 5)]))
    assert produced <= set(RECONSTRUCTED_FEATURES)


# --------------------------------------------------- which sessions are usable


def test_the_current_session_is_excluded_until_it_has_closed_and_cleared_the_delay():
    """Two independent reasons, either sufficient. A session still trading has
    not settled, and settlement is what labels it. And the complimentary feed
    refuses the current session's tape, reporting it as an unsigned OPRA
    agreement, which is not what the refusal means."""
    zone = ZoneInfo("America/New_York")
    close = time(16, 0)
    day = date(2026, 8, 31)

    def at(hour, minute=0):
        return reconstruct.last_rebuildable_session(
            datetime.combine(day, time(hour, minute), tzinfo=zone), close, zone
        )

    # Before the open, mid-session, and at the bell: yesterday is the latest.
    assert at(5, 23) == date(2026, 8, 30)
    assert at(11, 0) == date(2026, 8, 30)
    assert at(16, 0) == date(2026, 8, 30)
    # Inside the feed's delay, still yesterday.
    assert at(16, 14) == date(2026, 8, 30)
    # Once the delay has passed, today is rebuildable.
    assert at(16, 15) == date(2026, 8, 31)
    assert at(20, 0) == date(2026, 8, 31)


def test_the_cutoff_is_computed_in_the_market_timezone_not_the_callers():
    """A UTC clock reading 02:00 on the first is still the previous afternoon
    in New York, and must not advance the window a day early."""
    zone = ZoneInfo("America/New_York")
    close = time(16, 0)
    utc_small_hours = datetime(2026, 9, 1, 2, 0, tzinfo=timezone.utc)
    assert reconstruct.last_rebuildable_session(utc_small_hours, close, zone) == date(2026, 8, 31)


EXPIRY = date(2026, 8, 31)
SPOT = 769.30


def _printed(strike: float, price: float, right: Right = Right.PUT) -> ReconstructedEntry:
    """One strike's print, with the fields the cleaning actually reads."""
    return ReconstructedEntry(
        contract=OptionContract(
            symbol=occ_symbol("SPY", EXPIRY, right, strike),
            underlying="SPY",
            right=right,
            strike=strike,
            expiry=EXPIRY,
            multiplier=100,
        ),
        entry_price=price,
        implied_volatility=None,
        trades=5,
        volume=50,
    )


def _strikes(rows) -> list[float]:
    return [row.contract.strike for row in rows]


@pytest.mark.parametrize(
    "right, base, prices",
    [
        (Right.PUT, 755.0, [0.40, 0.90, 2.10, 4.60]),
        (Right.CALL, 770.0, [4.60, 2.10, 0.90, 0.40]),
    ],
)
def test_a_ladder_a_real_book_could_have_shown_survives_untouched(right, base, prices):
    """Cleaning is only worth having if it leaves an honest chain alone."""
    rows = [_printed(base + 5.0 * i, price, right) for i, price in enumerate(prices)]
    assert _strikes(arbitrage_free(rows, SPOT)) == [base + 5.0 * i for i in range(4)]


def test_a_vertical_costing_more_than_its_width_does_not_survive():
    """The 765/770 put spread here costs 8.00 to own 5.00 of strike distance,
    which no book has ever quoted: the most it can pay at expiry is 5.00."""
    rows = [_printed(755.0, 1.0), _printed(765.0, 16.0), _printed(770.0, 24.0)]
    assert 770.0 not in _strikes(arbitrage_free(rows, SPOT))


def test_a_convex_surface_can_still_price_a_broken_wing_at_an_impossible_credit():
    """The condition convexity misses, and the reason it misses it.

    Convexity bounds the butterfly weighted by the wing widths. A broken wing
    is weighted 1/-2/1, and the two agree only when the wings are equal. So
    this ladder is convex, monotone and within bounds, and still pays the
    structure a credit of 7.00 against wings 5.00 and 10.00 apart, which is
    2.00 of profit at every price the underlying can expire at.
    """
    upper, body, lower = _printed(770.0, 24.0), _printed(765.0, 16.0), _printed(755.0, 1.0)
    near_wing, far_wing = 770.0 - 765.0, 765.0 - 755.0

    credit = -(upper.entry_price - 2 * body.entry_price + lower.entry_price)
    assert credit > far_wing - near_wing

    # Convex: the body sits under the chord joining the wings.
    weight = (770.0 - 765.0) / (770.0 - 755.0)
    assert body.entry_price <= weight * lower.entry_price + (1 - weight) * upper.entry_price

    # And the cleaning refuses it anyway, on the vertical rather than the
    # butterfly. Both offending prints go, leaving nothing to build it from.
    survivors = _strikes(arbitrage_free([lower, body, upper], SPOT))
    assert 770.0 not in survivors and 765.0 not in survivors
