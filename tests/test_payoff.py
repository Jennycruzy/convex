"""Closed-form checks on the risk calculator that gates every order."""

from __future__ import annotations

import pytest

from convex.errors import UndefinedRiskError
from convex.instruments import Right
from convex.payoff import max_loss, payoff_at, payoff_curve, risk_profile, upside_slope
from tests.conftest import leg


def test_put_bwb_loss_is_the_broken_wing_width_less_the_credit(put_bwb_legs):
    # Wings are 5 wide above and 10 wide below, so the payoff below the lowest
    # strike is -5 per share before cash, and the structure is entered here for
    # a 0.20 credit: 650 - 2*645 + 635 = -5, less a credit of 0.20, is 4.80.
    credit = -0.20
    loss, price = max_loss(put_bwb_legs, credit)
    assert loss == pytest.approx(480.0)
    assert price == 635.0
    assert payoff_at(put_bwb_legs, credit, 700.0) == pytest.approx(20.0)


def test_put_bwb_upside_is_riskless_when_entered_for_credit(put_bwb_legs):
    profile = risk_profile(put_bwb_legs, -0.20)
    assert profile.is_credit
    assert payoff_at(put_bwb_legs, -0.20, 660.0) > 0.0
    assert profile.max_profit == pytest.approx(520.0)
    assert profile.max_profit_price == pytest.approx(645.0)
    # A credit broken-wing butterfly crosses zero once, on the way down. Above
    # that single breakeven the payoff never returns below zero, which is the
    # riskless tail the dashboard draws.
    assert profile.breakevens == (639.8,)


def test_debit_butterfly_loss_is_the_debit(put_bwb_legs):
    symmetric = [
        leg(650.0, Right.PUT, 3.00, 3.10, +1),
        leg(645.0, Right.PUT, 1.60, 1.70, -2),
        leg(640.0, Right.PUT, 0.80, 0.90, +1),
    ]
    loss, _ = max_loss(symmetric, 0.35)
    assert loss == pytest.approx(35.0)


def test_uncovered_short_call_is_refused():
    naked = [leg(660.0, Right.CALL, 1.00, 1.10, -1)]
    assert upside_slope(naked) == -1
    with pytest.raises(UndefinedRiskError, match="unbounded"):
        max_loss(naked, -1.00)


def test_ratio_spread_without_a_wing_is_refused_on_the_downside_too():
    # A raw 1x2 put ratio spread has bounded loss only because the underlying
    # cannot go below zero, and that bound is the whole account. The calculator
    # returns it rather than raising, and the risk budget is what rejects it.
    ratio = [
        leg(650.0, Right.PUT, 3.00, 3.10, +1),
        leg(645.0, Right.PUT, 1.60, 1.70, -2),
    ]
    loss, price = max_loss(ratio, -0.20)
    assert price == 0.0  # the only bound is that SPY cannot trade below zero
    assert loss == pytest.approx(64_000.0 - 20.0)


def test_apparent_arbitrage_is_refused_rather_than_sized():
    free_money = [
        leg(650.0, Right.PUT, 3.00, 3.10, +1),
        leg(645.0, Right.PUT, 1.60, 1.70, -2),
        leg(630.0, Right.PUT, 0.20, 0.25, +1),
    ]
    # Wings of 5 and 20 floor the payoff at -15 per share, so a quoted credit
    # larger than that is a free lunch and therefore a bad quote.
    with pytest.raises(UndefinedRiskError, match="arbitrage"):
        max_loss(free_money, -16.00)


def test_payoff_curve_includes_every_kink(put_bwb_legs):
    curve = payoff_curve(put_bwb_legs, -0.20, 600.0, 700.0, points=11)
    prices = [price for price, _ in curve]
    for strike in (635.0, 645.0, 650.0):
        assert strike in prices
