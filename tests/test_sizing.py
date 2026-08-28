"""Sizing is a function of the worst case and the tail, and of nothing else."""

from __future__ import annotations

from datetime import date, datetime, time, timezone

import numpy as np
import pytest

from convex.config import load
from convex.costs import CostModel
from convex.edge import evaluate
from convex.errors import ConfigError
from convex.scenarios import ScenarioSet
from convex.sizing import PortfolioState, size_position
from convex.structures.base import chain_index
from convex.structures.builders import put_broken_wing_butterflies


@pytest.fixture
def config():
    return load()


@pytest.fixture
def estimate(test_chain, config):
    grid = np.linspace(-0.02, 0.015, 200)
    scenarios = ScenarioSet(
        log_returns=grid,
        source_days=tuple(date(2026, 1, 1) for _ in grid),
        entry_time=time(10, 0),
        exit_time=time(16, 0),
        volatility_scale=1.0,
        built_at=datetime.now(timezone.utc),
    )
    candidate = put_broken_wing_butterflies(chain_index(test_chain), config, 650.0)[-1]
    return evaluate(candidate.legs, scenarios, CostModel.from_config(config), 650.0, 1, 0.01)


def test_size_never_risks_more_than_the_configured_fraction(estimate, config):
    portfolio = PortfolioState(equity=100_000.0, buying_power=200_000.0, open_structures=0, es_in_use=0.0)
    decision = size_position(estimate, portfolio, config)
    risked = decision.contracts * estimate.profile.max_loss
    assert risked <= portfolio.equity * config.float_("risk.risk_pct_per_structure")


def test_the_tail_can_bind_before_the_risk_budget(estimate, config):
    exhausted = PortfolioState(
        equity=100_000.0,
        buying_power=200_000.0,
        open_structures=3,
        es_in_use=100_000.0 * config.float_("risk.portfolio_es_cap_pct"),
    )
    decision = size_position(estimate, exhausted, config)
    assert decision.contracts == 0
    assert decision.binding_constraint == "portfolio_tail"


def test_buying_power_can_bind(estimate, config):
    thin = PortfolioState(equity=100_000.0, buying_power=1.0, open_structures=0, es_in_use=0.0)
    decision = size_position(estimate, thin, config)
    assert decision.contracts == 0
    assert decision.binding_constraint == "buying_power"


def test_sizing_refuses_a_multi_lot_estimate(test_chain, config):
    grid = np.linspace(-0.01, 0.01, 50)
    scenarios = ScenarioSet(
        log_returns=grid,
        source_days=tuple(date(2026, 1, 1) for _ in grid),
        entry_time=time(10, 0),
        exit_time=time(16, 0),
        volatility_scale=1.0,
        built_at=datetime.now(timezone.utc),
    )
    candidate = put_broken_wing_butterflies(chain_index(test_chain), config, 650.0)[0]
    multi = evaluate(candidate.legs, scenarios, CostModel.from_config(config), 650.0, 5, 0.01)
    portfolio = PortfolioState(equity=100_000.0, buying_power=200_000.0, open_structures=0, es_in_use=0.0)
    with pytest.raises(ConfigError, match="single-lot"):
        size_position(multi, portfolio, config)
