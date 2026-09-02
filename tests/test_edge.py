"""Edge, cost and tail arithmetic, and the ranking they feed."""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import numpy as np
import pytest

from convex.config import load
from convex.costs import CostModel
from convex.edge import at_limit, evaluate, expected_shortfall
from convex.instruments import Right
from convex.scenarios import ScenarioSet
from convex.structures.base import chain_index
from convex.structures.builders import debit_verticals, put_broken_wing_butterflies
from tests.conftest import leg


@pytest.fixture
def config():
    return load()


@pytest.fixture
def scenarios():
    """A hundred equally weighted session returns with a downside tail."""
    grid = np.linspace(-0.03, 0.02, 100)
    return ScenarioSet(
        log_returns=grid,
        source_days=tuple(date(2026, 1, 1) for _ in grid),
        entry_time=time(10, 0),
        exit_time=time(16, 0),
        volatility_scale=1.0,
        built_at=datetime.now(UTC),
    )


def test_expected_shortfall_is_the_mean_of_the_worst_tail():
    outcomes = np.array([-100.0, -50.0, 0.0, 10.0, 20.0])
    assert expected_shortfall(outcomes, 0.4) == pytest.approx(75.0)
    # A tail thinner than one scenario still reports the single worst outcome
    # rather than collapsing to zero.
    assert expected_shortfall(outcomes, 0.01) == pytest.approx(100.0)


def test_net_edge_is_gross_less_every_charge(test_chain, config, scenarios):
    model = CostModel.from_config(config)
    candidate = put_broken_wing_butterflies(chain_index(test_chain), config, 650.0)[0]
    estimate = evaluate(candidate.legs, scenarios, model, 650.0, 1, 0.01)

    waterfall = estimate.waterfall()
    rebuilt = (
        waterfall["gross_edge"]
        + waterfall["half_spread"]
        + waterfall["slippage"]
        + waterfall["fees"]
        + waterfall["exit_reserve"]
    )
    assert rebuilt == pytest.approx(waterfall["net_edge"], abs=0.02)
    assert estimate.cost.total > 0.0
    assert estimate.expected_shortfall > 0.0


def test_cost_scales_with_leg_count(test_chain, config, scenarios):
    model = CostModel.from_config(config)
    index = chain_index(test_chain)
    four_legged = put_broken_wing_butterflies(index, config, 650.0)[0]
    two_legged = debit_verticals(index, config, 650.0)[0]

    fly = evaluate(four_legged.legs, scenarios, model, 650.0, 1, 0.01)
    vertical = evaluate(two_legged.legs, scenarios, model, 650.0, 1, 0.01)
    assert fly.cost.total > vertical.cost.total
    assert fly.cost.leg_count > vertical.cost.leg_count


def test_tail_sizing_includes_the_short_leg_exit_reserve(test_chain, config, scenarios):
    """The loss budget must see the same all-in friction as net edge does."""
    model = CostModel.from_config(config)
    candidate = put_broken_wing_butterflies(chain_index(test_chain), config, 650.0)[0]
    estimate = evaluate(candidate.legs, scenarios, model, 650.0, 1, 0.01)

    assert model.risk_debit(candidate.legs) > model.executable_debit(candidate.legs)
    assert estimate.profile.net_entry_debit == pytest.approx(model.risk_debit(candidate.legs))


def test_a_wide_spread_can_turn_a_positive_gross_edge_negative(config):
    """The finding the whole project is built on, reproduced in miniature.

    A structure with a real but modest gross edge, quoted at spreads a judge
    would recognise as ordinary for SPY 0DTE wings, is unprofitable once the
    four crossings are paid for. Nothing about the payoff shape changed.
    """
    model = CostModel.from_config(config)
    grid = np.linspace(-0.0025, -0.0015, 100)
    scenarios = ScenarioSet(
        log_returns=grid,
        source_days=tuple(date(2026, 1, 1) for _ in grid),
        entry_time=time(10, 0),
        exit_time=time(16, 0),
        volatility_scale=1.0,
        built_at=datetime.now(UTC),
    )
    wide = [
        leg(650.0, Right.PUT, 2.50, 3.60, +1),
        leg(645.0, Right.PUT, 1.20, 2.10, -2),
        leg(635.0, Right.PUT, 0.10, 0.90, +1),
    ]
    estimate = evaluate(wide, scenarios, model, 650.0, 1, 0.01)
    assert estimate.gross_edge > 0.0
    assert estimate.net_edge < 0.0
    assert estimate.cost_share_of_gross > 1.0


def test_a_worse_limit_reduces_edge_and_increases_the_tail(test_chain, config, scenarios):
    model = CostModel.from_config(config)
    candidate = put_broken_wing_butterflies(chain_index(test_chain), config, 650.0)[0]
    estimate = evaluate(candidate.legs, scenarios, model, 650.0, 1, 0.01)
    worse_limit = model.executable_debit(candidate.legs) + 0.02

    repriced = at_limit(estimate, candidate.legs, model, worse_limit, 0.01)

    assert repriced.net_edge == pytest.approx(estimate.net_edge - 2.0)
    assert repriced.profile.max_loss == pytest.approx(estimate.profile.max_loss + 2.0)
    assert repriced.expected_shortfall >= estimate.expected_shortfall
