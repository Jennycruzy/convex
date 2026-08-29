"""Every gate is observed rejecting something. A gate that has never fired
in a test does not exist, so each of these makes one fire on purpose."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from convex.config import load
from convex.errors import ConfigError
from convex.costs import CostModel
from convex.edge import evaluate
from convex.gates import (
    ALL_GATES,
    GateContext,
    LiquidityGate,
    run_candidate_gates,
    run_session_gates,
)
from convex.instruments import Right
from convex.scenarios import ScenarioSet
from convex.sizing import PortfolioState, size_position
from convex.structures.base import chain_index
from convex.structures.builders import put_broken_wing_butterflies
from tests.conftest import leg

# The chain fixture quotes are stamped when they are built, so the gate clock
# has to be the same clock for quote ages to mean anything.
NOW = datetime.now(timezone.utc)


@pytest.fixture
def config():
    return load()


@pytest.fixture
def measured(config):
    """The configuration as it stands once calibration has replaced the guesses.

    A session test that is about some other gate uses this, so the calibration
    check standing down cannot mask the gate actually under test.
    """
    return type(config)(
        path=config.path,
        loaded_mtime=config.loaded_mtime,
        values={**config.values, "provenance": {"hypothesis": []}},
    )


@pytest.fixture
def scenarios():
    grid = np.linspace(-0.02, 0.015, 200)
    return ScenarioSet(
        log_returns=grid,
        source_days=tuple(date(2026, 1, 1) for _ in grid),
        entry_time=time(10, 0),
        exit_time=time(16, 0),
        volatility_scale=1.0,
        built_at=NOW,
    )


@pytest.fixture
def skewed_scenarios():
    """A session distribution the chain is not pricing.

    The neutral fixture is deliberately close to the volatility the test chain
    was priced at, so nothing has an edge under it. Here the body of the
    distribution sits half a percent lower with a heavier left tail, which is
    the physical-versus-risk-neutral gap a put structure is meant to harvest.
    """
    body = np.linspace(-0.009, 0.002, 180)
    tail = np.linspace(-0.035, -0.012, 20)
    return ScenarioSet(
        log_returns=np.concatenate([tail, body]),
        source_days=tuple(date(2026, 1, 1) for _ in range(200)),
        entry_time=time(10, 0),
        exit_time=time(16, 0),
        volatility_scale=1.0,
        built_at=NOW,
    )


@pytest.fixture
def priced_candidates(test_chain, config, skewed_scenarios):
    """Every put broken-wing butterfly on the chain, priced net of cost."""
    model = CostModel.from_config(config)
    built = put_broken_wing_butterflies(chain_index(test_chain), config, 650.0)
    return [
        (subject, evaluate(subject.legs, skewed_scenarios, model, 650.0, 1, 0.01))
        for subject in built
    ]


@pytest.fixture
def best(priced_candidates, config, tmp_path):
    """The structure the ranking would actually pick.

    Deep wings on a one-dollar strike grid are quoted in pennies, so their
    relative spread is enormous and the liquidity gate throws most of them out
    before edge is even compared. Ranking on net edge alone would pick one of
    those, which is why the pipeline filters first and ranks second.
    """
    ctx = context(config, tmp_path)
    liquidity = LiquidityGate()
    tradable = [
        pair
        for pair in priced_candidates
        if liquidity.check(ctx, pair[0], pair[1], None).passed
    ]
    assert tradable, "the whole chain was rejected on liquidity"
    assert len(tradable) < len(priced_candidates), "no candidate was rejected on liquidity"
    return max(tradable, key=lambda pair: pair[1].net_edge)


@pytest.fixture
def candidate(best):
    return best[0]


@pytest.fixture
def estimate(best):
    return best[1]


def context(config, tmp_path: Path, **overrides) -> GateContext:
    base = dict(
        config=config,
        now_exchange=NOW,
        session_close=NOW + timedelta(hours=6),
        is_trading_day=True,
        market_open=True,
        equity=100_000.0,
        last_equity=100_000.0,
        buying_power=200_000.0,
        spot=650.0,
        open_structures=0,
        es_in_use=0.0,
        cumulative_fees=0.0,
        kill_switch_path=tmp_path / "KILL",
        probability=0.62,
    )
    base.update(overrides)
    return GateContext(**base)


def test_all_gates_are_named_and_distinct():
    names = [gate.name for gate in ALL_GATES]
    assert len(names) == len(set(names))
    assert {"net_of_cost", "assignment", "kill_switch", "expected_shortfall"} <= set(names)


def test_nothing_clears_the_hurdle_when_the_chain_prices_the_distribution(
    test_chain, config, scenarios, tmp_path
):
    """When the physical distribution matches what the chain implies, there is
    no edge to harvest and every candidate is refused on cost. This is the
    stand-down case, and it is the common one."""
    model = CostModel.from_config(config)
    ctx = context(config, tmp_path)
    liquidity = LiquidityGate()
    built, priced = [], []
    for subject in put_broken_wing_butterflies(chain_index(test_chain), config, 650.0):
        estimate = evaluate(subject.legs, scenarios, model, 650.0, 1, 0.01)
        # Penny wings are priced at the chain's floor rather than by the model,
        # so they are excluded here for the same reason the pipeline excludes
        # them: the liquidity gate throws them out before edge is compared.
        if liquidity.check(ctx, subject, estimate, None).passed:
            built.append(subject)
            priced.append(estimate)
    assert built
    assert max(estimate.net_edge for estimate in priced) < 0.0
    assert not any(
        run_candidate_gates(
            ctx, c, e, size_position(e, PortfolioState(100_000.0, 200_000.0, 0, 0.0), config)
        ).passed
        for c, e in zip(built, priced)
    )


def test_session_gates_pass_on_an_ordinary_open(measured, tmp_path):
    assert run_session_gates(context(measured, tmp_path)).passed


def test_kill_switch_stops_the_session(config, tmp_path):
    switch = tmp_path / "KILL"
    switch.write_text("halted by hand\n")
    report = run_session_gates(context(config, tmp_path, kill_switch_path=switch))
    assert not report.passed
    assert report.first_failure.name == "kill_switch"


def test_daily_loss_limit_stops_the_session(measured, tmp_path):
    report = run_session_gates(context(measured, tmp_path, equity=96_000.0))
    assert not report.passed
    assert {f.name for f in report.failures} == {"daily_loss_limit"}


def test_closed_day_and_submission_cutoff_both_stop_the_session(config, tmp_path):
    assert not run_session_gates(context(config, tmp_path, is_trading_day=False)).passed
    late = context(config, tmp_path, submission_cutoff=NOW - timedelta(minutes=1))
    assert not run_session_gates(late).passed


def test_cost_budget_stops_the_session(measured, tmp_path):
    report = run_session_gates(context(measured, tmp_path, cumulative_fees=2_500.0))
    assert not report.passed
    assert report.first_failure.name == "cost_budget"


def test_candidate_gates_pass_on_a_reasonable_structure(config, tmp_path, candidate, estimate):
    ctx = context(config, tmp_path)
    size = size_position(estimate, PortfolioState(100_000.0, 200_000.0, 0, 0.0), config)
    report = run_candidate_gates(ctx, candidate, estimate, size)
    assert report.passed, [f.detail for f in report.failures]


def test_the_net_of_cost_gate_rejects_a_structure_cost_has_eaten(config, tmp_path, scenarios):
    wide = [
        leg(650.0, Right.PUT, 2.50, 3.60, +1),
        leg(645.0, Right.PUT, 1.20, 2.10, -2),
        leg(635.0, Right.PUT, 0.10, 0.90, +1),
    ]
    from convex.structures.base import Candidate, Family

    priced = evaluate(wide, scenarios, CostModel.from_config(config), 650.0, 1, 0.01)
    subject = Candidate(family=Family.PUT_BWB, legs=tuple(wide), description="wide test wings")
    size = size_position(priced, PortfolioState(100_000.0, 200_000.0, 0, 0.0), config)
    report = run_candidate_gates(context(config, tmp_path), subject, priced, size)
    assert "net_of_cost" in {failure.name for failure in report.failures}


def test_liquidity_rejects_a_leg_quoted_too_wide(config, tmp_path, scenarios):
    from convex.structures.base import Candidate, Family

    illiquid = [
        leg(650.0, Right.PUT, 0.10, 3.00, +1),
        leg(645.0, Right.PUT, 1.20, 1.30, -2),
        leg(635.0, Right.PUT, 0.30, 0.40, +1),
    ]
    priced = evaluate(illiquid, scenarios, CostModel.from_config(config), 650.0, 1, 0.01)
    subject = Candidate(family=Family.PUT_BWB, legs=tuple(illiquid), description="illiquid wing")
    size = size_position(priced, PortfolioState(100_000.0, 200_000.0, 0, 0.0), config)
    report = run_candidate_gates(context(config, tmp_path), subject, priced, size)
    assert "liquidity" in {failure.name for failure in report.failures}


def test_assignment_gate_refuses_an_in_the_money_short_near_the_close(
    config, tmp_path, candidate, estimate
):
    ctx = context(
        config,
        tmp_path,
        session_close=NOW + timedelta(minutes=10),
        spot=min(candidate.strikes) - 1.0,  # every put short leg is in the money
    )
    size = size_position(estimate, PortfolioState(100_000.0, 200_000.0, 0, 0.0), config)
    report = run_candidate_gates(ctx, candidate, estimate, size)
    assert "assignment" in {failure.name for failure in report.failures}


def test_confidence_gate_stands_down_on_a_coin_flip(config, tmp_path, candidate, estimate):
    ctx = context(config, tmp_path, probability=0.505)
    size = size_position(estimate, PortfolioState(100_000.0, 200_000.0, 0, 0.0), config)
    report = run_candidate_gates(ctx, candidate, estimate, size)
    assert "classifier_confidence" in {failure.name for failure in report.failures}


def test_staleness_gate_refuses_an_old_chain(config, tmp_path, candidate, estimate):
    ctx = context(config, tmp_path, now_exchange=NOW + timedelta(minutes=5))
    size = size_position(estimate, PortfolioState(100_000.0, 200_000.0, 0, 0.0), config)
    report = run_candidate_gates(ctx, candidate, estimate, size)
    assert "feature_staleness" in {failure.name for failure in report.failures}


def test_concurrency_and_tail_caps_both_bind(config, tmp_path, candidate, estimate):
    size = size_position(estimate, PortfolioState(100_000.0, 200_000.0, 0, 0.0), config)
    full = context(config, tmp_path, open_structures=4, es_in_use=3_000.0)
    failures = {f.name for f in run_candidate_gates(full, candidate, estimate, size).failures}
    assert "concurrency" in failures
    assert "expected_shortfall" in failures


# ------------------------------------------------- calibration provenance


def test_the_hypothesis_list_matches_the_comments_that_mark_them(config):
    """The list the code reads and the markings a reader sees must agree.

    Drift between them is the failure this pair is meant to prevent: a value
    silently dropped from the list would stop being guarded while still reading
    as unmeasured to anyone opening the file.
    """
    marked = set()
    for line in config.path.read_text().splitlines():
        if "HYPOTHESIS" not in line or ":" not in line:
            continue
        key = line.split(":", 1)[0].strip()
        if key.startswith("#"):
            continue
        marked.add(key)

    listed = {key.split(".")[-1] for key in config.hypotheses()}
    assert marked <= listed, f"marked HYPOTHESIS but not listed: {marked - listed}"


def test_a_hypothesis_naming_a_key_that_does_not_exist_raises(config):
    broken = type(config)(
        path=config.path,
        loaded_mtime=config.loaded_mtime,
        values={**config.values, "provenance": {"hypothesis": ["costs.no_such_key"]}},
    )
    with pytest.raises(ConfigError, match="no_such_key"):
        broken.hypotheses()


def test_the_session_stands_down_while_a_cost_input_is_still_unmeasured(
    config, tmp_path
):
    """Shipped state: the fees are zero and have never been measured."""
    report = run_session_gates(context(config, tmp_path))
    failure = report.first_failure
    assert failure is not None
    assert failure.name == "calibration"
    assert "per_contract_fee" in failure.detail


def test_the_session_proceeds_once_every_input_has_been_measured(measured, tmp_path):
    report = run_session_gates(context(measured, tmp_path))
    assert report.passed, [f.detail for f in report.failures]
