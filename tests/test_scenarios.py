"""The terminal distribution, and the horizon its variance history belongs to."""

from __future__ import annotations

from datetime import date, datetime, time, timezone

import numpy as np
import pytest

from convex.classifier import RegimeRule
from convex.errors import DataError
from convex.scenarios import ScenarioSet


def scenarios(returns: np.ndarray) -> ScenarioSet:
    return ScenarioSet(
        log_returns=returns,
        source_days=tuple(date(2026, 1, 1) for _ in returns),
        entry_time=time(10, 0),
        exit_time=time(16, 0),
        volatility_scale=1.0,
        built_at=datetime.now(timezone.utc),
    )


def test_the_window_is_the_one_the_returns_were_measured_over():
    """Six hours of a 365 day year, not the caller's remaining time."""
    assert scenarios(np.linspace(-0.01, 0.01, 20)).session_tau == pytest.approx(
        6.0 / (365.0 * 24.0)
    )


def test_a_window_that_does_not_run_forwards_is_refused():
    backwards = ScenarioSet(
        log_returns=np.linspace(-0.01, 0.01, 20),
        source_days=tuple(date(2026, 1, 1) for _ in range(20)),
        entry_time=time(16, 0),
        exit_time=time(10, 0),
        volatility_scale=1.0,
        built_at=datetime.now(timezone.utc),
    )
    with pytest.raises(DataError, match="not a positive span"):
        backwards.session_tau


def test_the_variance_history_does_not_move_with_the_time_of_day():
    """The regression that cost the 31 August session.

    The history used to be annualised against whatever time to the close the
    caller happened to have. Running the cycle at 13:22 rather than at 10:00
    divided full session returns by a third of the span they cover, inflating
    every reading and dragging today's implied variance down its own
    distribution. The rule then reported no view, on a chain that had one.

    A history built from the same returns has to describe the same market at
    any hour, and only the implied reading it is compared against may move.
    """
    returns = np.linspace(-0.02, 0.02, 200)
    history = scenarios(returns).annualised_variance()
    assert history == pytest.approx((returns**2) / (6.0 / (365.0 * 24.0)))

    rule = RegimeRule()
    implied = float(np.quantile(history, 0.75))
    assert rule.regime(implied, history) == "high_variance"
    # The same call, made from a set rebuilt at any other moment of the day,
    # because nothing about it reads a clock.
    assert rule.regime(implied, scenarios(returns).annualised_variance()) == "high_variance"
