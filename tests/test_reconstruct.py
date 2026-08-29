"""Rebuilding a past session, and refusing to invent the parts that are gone.

The arithmetic here is checked against itself, in that a price produced by the
pricer must solve back to the volatility that produced it, and the refusals are
checked explicitly, because the failure mode that matters is not a wrong number
but a confident one where there should be none.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from convex.config import load
from convex.errors import DataError
from convex.instruments import Right
from convex.reconstruct import (
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
