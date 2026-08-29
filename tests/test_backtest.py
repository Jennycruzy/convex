"""Replaying sessions, gross and net.

The statistic that matters here is the pair, not either half. These tests fix
the arithmetic against hand-computed answers and, most importantly, check that
a series which looks good gross and bad net is reported as exactly that rather
than being smoothed into one summary number.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from convex import backtest
from convex.errors import DataError
from convex.structures.base import Family
from convex.training import Sample

DAY = date(2026, 8, 3)


def sample(offset: int, gross: float, cost: float, family=Family.PUT_BWB) -> Sample:
    return Sample(
        session_date=DAY + timedelta(days=offset),
        family=family,
        features={},
        label=1 if gross - cost > 0 else 0,
        gross_pnl=gross,
        cost=cost,
        net_pnl=round(gross - cost, 2),
        description="test",
    )


# ------------------------------------------------------------------ statistics


def test_sharpe_of_a_constant_series_is_undefined_not_infinite():
    assert backtest.sharpe(np.array([5.0, 5.0, 5.0])) is None


def test_sharpe_of_a_single_session_is_undefined():
    assert backtest.sharpe(np.array([5.0])) is None


def test_a_sample_too_small_to_support_a_sharpe_does_not_get_one():
    """Ten near-identical trades produce a ratio in the hundreds.

    That is a small denominator, not a good strategy, and printing it would be
    the single most misleading number this project could publish.
    """
    tiny = np.array([9.0, 9.1, 8.9, 9.0, 9.05, 8.95, 9.02, 9.01, 8.99, 9.0])
    assert tiny.size < backtest.MINIMUM_OBSERVATIONS_FOR_SHARPE
    assert backtest.sharpe(tiny) is None


def test_sharpe_matches_the_hand_computation():
    series = np.random.default_rng(1).normal(size=40)
    expected = series.mean() / series.std(ddof=1) * np.sqrt(252)
    assert backtest.sharpe(series) == pytest.approx(expected)


def test_max_drawdown_is_the_worst_peak_to_trough_of_the_cumulative_curve():
    # cumulative: 10, 4, 9, 1 → peak 10, trough 1 → drawdown 9
    assert backtest.max_drawdown(np.array([10.0, -6.0, 5.0, -8.0])) == pytest.approx(9.0)
    assert backtest.max_drawdown(np.array([])) == 0.0


def test_a_series_that_only_rises_has_no_drawdown():
    assert backtest.max_drawdown(np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_expected_shortfall_averages_the_worst_tail_as_a_positive_loss():
    series = np.array([-50.0, -10.0, 5.0, 20.0])
    # 25% of four sessions is the single worst.
    assert backtest.expected_shortfall(series, 0.25) == pytest.approx(50.0)


def test_an_impossible_confidence_raises():
    with pytest.raises(DataError):
        backtest.expected_shortfall(np.array([1.0]), 1.5)


# ---------------------------------------------------------- the gross-net gap


def test_a_series_that_is_positive_gross_and_negative_net_is_reported_as_both():
    """The finding the whole project rests on, measured on our own replay."""
    rows = [sample(index, gross=10.0 if index % 2 else 4.0, cost=8.0) for index in range(40)]
    report = backtest.run(rows, probabilities={})
    arm = report.per_family[str(Family.PUT_BWB)]["every session"]

    assert arm.gross_total > 0, "this series does make money before costs"
    assert arm.net_total < 0, "and loses after them"
    assert arm.gross_sharpe > 0
    assert arm.net_sharpe < 0
    assert not arm.survives_costs
    assert arm.cost_share_of_gross > 1.0


def test_cost_share_is_undefined_rather_than_negative_when_gross_lost_money():
    rows = [sample(index, gross=-5.0, cost=2.0) for index in range(6)]
    arm = backtest.run(rows, probabilities={}).per_family[str(Family.PUT_BWB)]["every session"]
    assert arm.cost_share_of_gross is None


def test_a_sample_whose_cost_exceeded_a_positive_gross_reports_a_share_above_one():
    assert sample(0, gross=10.0, cost=14.0).cost_share == pytest.approx(1.4)


def test_cost_share_of_a_losing_sample_is_infinite_not_a_quiet_zero():
    assert sample(0, gross=-3.0, cost=2.0).cost_share == float("inf")


# ------------------------------------------------------------- hard mapping


def test_only_sessions_the_classifier_flagged_enter_the_classified_arm():
    rows = [sample(index, gross=10.0, cost=1.0) for index in range(6)]
    flagged = {row.session_date for row in rows[:2]}
    probabilities = {
        Family.PUT_BWB: {
            row.session_date: (0.7 if row.session_date in flagged else 0.3) for row in rows
        }
    }
    report = backtest.run(rows, probabilities)
    arms = report.per_family[str(Family.PUT_BWB)]
    assert arms["every session"].trades == 6
    assert arms["classified"].trades == 2


def test_a_session_inside_the_burn_in_is_not_traded_on_a_model_that_did_not_exist():
    rows = [sample(index, gross=10.0, cost=1.0) for index in range(4)]
    # No probability at all for the first two sessions.
    probabilities = {Family.PUT_BWB: {rows[2].session_date: 0.9, rows[3].session_date: 0.9}}
    arms = backtest.run(rows, probabilities).per_family[str(Family.PUT_BWB)]
    assert arms["classified"].trades == 2


def test_a_probability_of_exactly_one_half_does_not_trade():
    rows = [sample(0, gross=10.0, cost=1.0)]
    probabilities = {Family.PUT_BWB: {rows[0].session_date: 0.5}}
    assert backtest.run(rows, probabilities).per_family[str(Family.PUT_BWB)][
        "classified"
    ].trades == 0


# ------------------------------------------------------------------- basket


def test_the_basket_sums_families_within_a_session_not_across_them():
    rows = [
        sample(0, gross=10.0, cost=1.0, family=Family.PUT_BWB),
        sample(0, gross=6.0, cost=1.0, family=Family.STRADDLE),
        sample(1, gross=-4.0, cost=1.0, family=Family.PUT_BWB),
    ]
    report = backtest.run(rows, probabilities={})
    basket = report.basket["every session"]
    assert report.sessions == 2
    assert basket.trades == 2, "two sessions, not three structures"
    # Session one holds both families: 9 + 5. Session two holds one: -5.
    assert basket.net_total == pytest.approx((9.0 + 5.0) + -5.0)


def test_an_empty_replay_produces_an_empty_report_rather_than_raising():
    report = backtest.run([], probabilities={})
    assert report.sessions == 0 and report.per_family == {} and report.basket == {}


def test_the_report_serialises_every_figure_in_both_forms():
    rows = [sample(index, gross=10.0, cost=8.0) for index in range(5)]
    payload = backtest.run(rows, probabilities={}).as_dict()
    arm = payload["per_family"][str(Family.PUT_BWB)]["every session"]
    assert {"gross_total", "net_total", "gross_sharpe", "net_sharpe"} <= set(arm)
