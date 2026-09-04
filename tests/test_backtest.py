"""Replaying sessions, gross and net.

The statistic that matters here is the pair, not either half. These tests fix
the arithmetic against hand-computed answers and, most importantly, check that
a series which looks good gross and bad net is reported as exactly that rather
than being smoothed into one summary number.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import numpy as np
import pytest

from convex import backtest
from convex.dashboard import charts
from convex.errors import DataError
from convex.structures.base import Family
from convex.training import Sample

DAY = date(2026, 8, 3)


def sample(
    offset: int,
    gross: float,
    cost: float,
    family=Family.PUT_BWB,
    label_gross: float | None = None,
    label_cost: float | None = None,
) -> Sample:
    """One row. ``gross`` and ``cost`` are the traded figures, which is what
    the replay reads; the label triple defaults to matching them, as it does
    when the labels are taken across a single candidate."""
    label_gross = gross if label_gross is None else label_gross
    label_cost = cost if label_cost is None else label_cost
    return Sample(
        session_date=DAY + timedelta(days=offset),
        family=family,
        features={},
        label=1 if label_gross - label_cost > 0 else 0,
        label_gross_pnl=label_gross,
        label_cost=label_cost,
        label_net_pnl=round(label_gross - label_cost, 2),
        traded_gross_pnl=gross,
        traded_cost=cost,
        traded_net_pnl=round(gross - cost, 2),
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


def test_the_replay_totals_the_traded_candidate_and_not_the_label():
    """The regression this pair of triples exists to prevent.

    Under ``label_top_k`` above one the label is the median across the top few
    ranked candidates, which is a trade the agent never opens. Every number the
    replay reports is money, so it reads the traded figures; totting up the
    label would report the earnings of a candidate nobody bought.
    """
    # Traded makes 3.00 a session. The label, shrunk over five candidates,
    # claims a 90.00 loss. Nothing about the second is the strategy's result.
    rows = [
        sample(index, gross=5.0, cost=2.0, label_gross=10.0, label_cost=100.0)
        for index in range(6)
    ]
    arm = backtest.run(rows, probabilities={}).per_family[str(Family.PUT_BWB)]["every session"]

    assert arm.net_total == pytest.approx(18.0)
    assert arm.gross_total == pytest.approx(30.0)
    assert arm.cost_total == pytest.approx(12.0)

    basket = backtest.run(rows, probabilities={}).basket["every session"]
    assert basket.net_total == pytest.approx(18.0)


def test_the_classified_arm_also_earns_the_traded_result():
    rows = [
        sample(index, gross=5.0, cost=2.0, label_gross=10.0, label_cost=100.0)
        for index in range(6)
    ]
    taken = {Family.PUT_BWB: {row.session_date: 0.9 for row in rows}}
    report = backtest.run(rows, probabilities=taken)

    assert report.per_family[str(Family.PUT_BWB)]["classified"].net_total == pytest.approx(18.0)
    assert report.basket["classified"].net_total == pytest.approx(18.0)


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


# ------------------------------------------------------------- the equity curve


def test_the_running_total_is_kept_not_just_the_drawdown_measured_from_it():
    """max_drawdown already walks this curve. Keeping it is what lets the page
    draw the shape the drawdown is a number about."""
    rows = [sample(index, gross=5.0, cost=2.0) for index in range(4)]
    arm = backtest.run(rows, probabilities={}).per_family[str(Family.PUT_BWB)]["every session"]

    assert arm.net_curve == pytest.approx((3.0, 6.0, 9.0, 12.0))
    assert arm.gross_curve == pytest.approx((5.0, 10.0, 15.0, 20.0))
    assert arm.net_curve[-1] == pytest.approx(arm.net_total)
    assert arm.gross_curve[-1] == pytest.approx(arm.gross_total)


def test_the_curve_survives_the_round_trip_through_the_report():
    rows = [sample(index, gross=5.0, cost=2.0) for index in range(3)]
    payload = backtest.run(rows, probabilities={}).as_dict()
    arm = payload["per_family"][str(Family.PUT_BWB)]["every session"]
    assert arm["net_curve"] == [3.0, 6.0, 9.0]
    assert arm["gross_curve"] == [5.0, 10.0, 15.0]


def test_the_curve_is_the_traded_result_like_every_other_total():
    rows = [
        sample(index, gross=5.0, cost=2.0, label_gross=10.0, label_cost=100.0)
        for index in range(3)
    ]
    arm = backtest.run(rows, probabilities={}).per_family[str(Family.PUT_BWB)]["every session"]
    assert arm.net_curve == pytest.approx((3.0, 6.0, 9.0))


def test_an_empty_equity_curve_is_a_sentence_not_a_flat_line_at_zero():
    """A flat line reads as a strategy that traded and broke even, which is a
    far more flattering claim than not having traded."""
    drawn = charts.equity_svg([], [])
    assert "<svg" not in drawn
    assert "no settled trades yet" in drawn.lower()

    one = charts.equity_svg([10.0], [5.0])
    assert "<svg" not in one
    assert "+5.00" in one


def test_the_equity_curve_draws_both_series_and_labels_its_denominator():
    drawn = charts.equity_svg(
        [10.0, 25.0, 18.0, 40.0], [6.0, 15.0, 4.0, 12.0],
        sessions=276, label="basket, classified",
    )
    assert "<svg" in drawn
    assert "+12.00 net" in drawn
    assert "+40.00 gross" in drawn
    assert "4 trades over 276 sessions" in drawn
    assert "basket, classified" in drawn


def test_mismatched_curve_lengths_raise_rather_than_draw_a_wrong_shape():
    with pytest.raises(ValueError):
        charts.equity_svg([1.0, 2.0], [1.0, 2.0, 3.0])


def test_a_curve_that_ends_under_water_is_coloured_as_a_loss():
    losing = charts.equity_svg([5.0, 9.0], [-2.0, -8.0])
    winning = charts.equity_svg([5.0, 9.0], [2.0, 8.0])
    assert "var(--down)" in losing and "var(--up)" not in losing
    assert "var(--up)" in winning


def _end_labels(drawn: str) -> list[tuple[float, str]]:
    """The two figures parked at the right edge, as (y, text).

    The axis ticks are right-anchored too, so they are filtered out by the one
    thing that separates them: an end label names its series.
    """
    found = re.findall(r"(<text[^>]*text-anchor='end'[^>]*>)([^<]*)</text>", drawn)
    return [
        (float(re.search(r"y='([\d.]+)'", tag).group(1)), text)
        for tag, text in found
        if " net" in text or " gross" in text or " before fees" in text
    ]


def test_two_series_finishing_together_do_not_print_one_label_over_the_other():
    """Both end labels hang off the same edge, so near-equal ends collide.

    The account's own chart finishes 6.62 apart on a range of about 1,200,
    which put the two figures 0.7 pixels apart and made both unreadable.
    """
    labels = _end_labels(
        charts.equity_svg([-560.0, -840.0, -1020.0], [-565.3, -846.62, -1026.62])
    )
    assert len(labels) == 2, labels
    assert abs(labels[0][0] - labels[1][0]) >= charts.LABEL_GAP


def test_series_finishing_far_apart_keep_their_own_positions():
    """The lift is only for a collision; a wide gap is left where it falls."""
    labels = _end_labels(charts.equity_svg([100.0, 16218.68], [50.0, 3375.53]))
    assert len(labels) == 2, labels
    assert abs(labels[0][0] - labels[1][0]) > charts.LABEL_GAP * 4
