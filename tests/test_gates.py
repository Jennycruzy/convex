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
    CalibrationGate,
    GateContext,
    LiquidityGate,
    NetOfCostGate,
    PositiveNetEdgeBoundGate,
    run_candidate_gates,
    run_session_gates,
)
from convex.instruments import Right
from convex.scenarios import ScenarioSet
from convex.sizing import PortfolioState, size_position
from convex.structures.base import Candidate, Family, chain_index
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
        values={
            **config.values,
            # Only the blocking list is emptied. The bounds stay exactly as
            # shipped, because a session running on deliberate over-estimates is
            # the normal case until a fill exists to measure them from.
            "provenance": {
                **config.values["provenance"],
                "hypothesis": [],
            },
        },
    )


@pytest.fixture
def unmeasured(config):
    """The configuration as it stands before a session has measured anything.

    Built rather than read. This used to be the shipped state of the file, so
    the test below read it straight off disk and passed; the first real
    calibration on 31 August wrote the threshold and cleared it from the
    blocking list, and the test broke on a working measurement. What is under
    test is the check, not which keys happen to be outstanding today.
    """
    return type(config)(
        path=config.path,
        loaded_mtime=config.loaded_mtime,
        values={
            **config.values,
            "provenance": {
                **config.values["provenance"],
                "hypothesis": ["liquidity.max_relative_spread"],
            },
        },
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


def test_closed_day_and_submission_cutoff_both_stop_the_session(measured, tmp_path):
    """Named, not merely failed.

    This ran against the unmeasured config, so the calibration check failed
    first and the assertion was satisfied whatever the calendar gate did. The
    gate is now the one asked, and it has to be the one that answers.
    """
    closed = run_session_gates(context(measured, tmp_path, is_trading_day=False))
    assert not closed.passed
    assert closed.first_failure.name == "market_calendar"

    late = run_session_gates(
        context(measured, tmp_path, submission_cutoff=NOW - timedelta(minutes=1))
    )
    assert not late.passed
    assert late.first_failure.name == "market_calendar"
    assert "cutoff" in late.first_failure.detail


def test_cost_budget_stops_the_session(measured, tmp_path):
    report = run_session_gates(context(measured, tmp_path, cumulative_fees=2_500.0))
    assert not report.passed
    assert report.first_failure.name == "cost_budget"


def test_candidate_gates_pass_on_a_reasonable_structure(config, tmp_path):
    """A tight, defined-risk BWB with enough gross edge clears every gate.

    The old shared chain fixture has three-cent half-spreads. After live
    calibration tightened the liquidity ceiling, every otherwise-liquid BWB in
    that fixture had its small gross edge consumed by realistic costs. That is
    correct behaviour, but it left this test without a passing control.
    """
    candidate = Candidate(
        family=Family.PUT_BWB,
        legs=(
            leg(652.0, Right.PUT, 3.00, 3.02, +1),
            leg(650.0, Right.PUT, 1.800, 1.816, -2),
            leg(647.0, Right.PUT, 0.400, 0.404, +1),
        ),
        description="tight put broken-wing butterfly",
    )
    favourable = ScenarioSet(
        log_returns=np.zeros(200),
        source_days=tuple(date(2026, 1, 1) for _ in range(200)),
        entry_time=time(10, 0),
        exit_time=time(16, 0),
        volatility_scale=1.0,
        built_at=NOW,
    )
    estimate = evaluate(candidate.legs, favourable, CostModel.from_config(config), 650.0, 1, 0.01)
    ctx = context(config, tmp_path)
    size = size_position(estimate, PortfolioState(100_000.0, 200_000.0, 0, 0.0), config)
    report = run_candidate_gates(ctx, candidate, estimate, size)
    assert report.passed, [f.detail for f in report.failures]


def test_the_net_of_cost_gate_rejects_a_nominal_remaining_edge(config, tmp_path):
    from types import SimpleNamespace

    report = NetOfCostGate().check(
        context(config, tmp_path),
        SimpleNamespace(),
        SimpleNamespace(
            net_edge=2.30,
            gross_edge=14.09,
            cost=SimpleNamespace(total=11.79, leg_count=3),
        ),
        None,
    )
    assert not report.passed
    assert report.threshold == 25.0
    assert "required minimum" in report.detail


def test_lower_confidence_bound_rejects_a_nominally_positive_edge(config, tmp_path):
    from types import SimpleNamespace

    report = PositiveNetEdgeBoundGate().check(
        context(config, tmp_path),
        SimpleNamespace(),
        SimpleNamespace(net_edge_lower_bound=lambda confidence: -12.34),
        None,
    )
    assert not report.passed
    assert report.threshold == 0.01
    assert "lower bound" in report.detail


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


def test_calibration_cannot_loosen_the_validated_admission_cap(
    config, tmp_path, candidate, estimate
):
    values = {
        **config.values,
        "liquidity": {
            **config.values["liquidity"],
            "max_relative_spread": 0.0513,
        },
    }
    observed_wider = type(config)(
        path=config.path, loaded_mtime=config.loaded_mtime, values=values
    )
    report = LiquidityGate().check(
        context(observed_wider, tmp_path), candidate, estimate, None
    )
    assert report.threshold == pytest.approx(0.01)
    assert "hard cap 1.0%" in report.detail


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


def test_the_bound_list_matches_the_comments_that_mark_them(config):
    """The same guard for the second list.

    A value marked BOUND but absent from conservative_bound reads to anyone
    opening the file as though it were still being watched, while the session
    says nothing about it at all.
    """
    marked = set()
    for line in config.path.read_text().splitlines():
        if "BOUND" not in line or ":" not in line:
            continue
        key = line.split(":", 1)[0].strip()
        if key.startswith("#"):
            continue
        marked.add(key)

    listed = {key.split(".")[-1] for key in config.bounds()}
    assert marked <= listed, f"marked BOUND but not listed: {marked - listed}"


def test_a_key_cannot_be_a_guess_and_a_bound_at_the_same_time(config):
    """One is unsafe to trade on and the other is not, so the two lists have to
    be disjoint or the gate's answer depends on which it happens to read."""
    overlap = set(config.hypotheses()) & set(config.bounds())
    assert not overlap, f"listed as both: {sorted(overlap)}"


def test_a_hypothesis_naming_a_key_that_does_not_exist_raises(config):
    broken = type(config)(
        path=config.path,
        loaded_mtime=config.loaded_mtime,
        values={
            **config.values,
            "provenance": {
                **config.values["provenance"],
                "hypothesis": ["costs.no_such_key"],
            },
        },
    )
    with pytest.raises(ConfigError, match="no_such_key"):
        broken.hypotheses()


def test_the_session_stands_down_while_a_cost_input_is_still_unmeasured(
    unmeasured, tmp_path
):
    """An unmeasured liquidity threshold stops the session.

    It blocks and the fee bounds do not, which is the whole point of the split.
    A threshold that is too permissive lets a bad leg through; a fee set above
    what it can be only refuses a candidate that was marginal anyway.
    """
    report = run_session_gates(context(unmeasured, tmp_path))
    failure = report.first_failure
    assert failure is not None
    assert failure.name == "calibration"
    assert "liquidity.max_relative_spread" in failure.detail
    assert "per_contract_fee" not in failure.detail


def test_the_session_proceeds_once_every_input_has_been_measured(measured, tmp_path):
    report = run_session_gates(context(measured, tmp_path))
    assert report.passed, [f.detail for f in report.failures]
    # Passing is not the same as being silent about it. The session says which
    # figures are still bounds rather than measurements.
    calibration = next(r for r in report.results if r.name == "calibration")
    assert "over-estimates" in calibration.detail
    # Every key the configuration still lists as a bound is named, and nothing
    # else is. Asserting the list rather than one hard-coded key means measuring
    # something and clearing it from the list cannot silently stop it being
    # reported, and cannot break this test either. costs.per_contract_fee used
    # to be named here and was measured on 2026-08-31 against the broker's own
    # fee activities, which is exactly the transition this now tolerates.
    bounds = [str(name) for name in measured.list_("provenance.conservative_bound")]
    assert bounds, "the fixture should still carry at least one standing bound"
    for name in bounds:
        assert name in calibration.detail
    for name in measured.list_("provenance.hypothesis"):
        assert str(name) not in calibration.detail


def test_a_bound_never_blocks_but_a_hypothesis_always_does(measured, tmp_path):
    """The asymmetry, asserted directly.

    Over-stating a cost can only refuse a trade that was marginally worth
    taking. Under-stating one admits a trade that was not, which is the failure
    the whole project is built to avoid, so only the second kind stops a
    session.
    """
    assert set(measured.bounds()) & set(CalibrationGate.REQUIRED)
    assert run_session_gates(context(measured, tmp_path)).passed

    guessing = type(measured)(
        path=measured.path,
        loaded_mtime=measured.loaded_mtime,
        values={
            **measured.values,
            "provenance": {
                **measured.values["provenance"],
                "hypothesis": ["costs.per_contract_fee"],
            },
        },
    )
    assert not run_session_gates(context(guessing, tmp_path)).passed


def test_a_structure_risking_more_than_its_budget_is_refused(
    test_chain, config, scenarios, tmp_path
):
    """The max-loss gate, observed refusing rather than assumed present.

    Law 9: a gate that has never been seen rejecting anything has not been
    demonstrated. This one had tests around it and none of them made it fire.
    """
    from convex.gates import MaxLossGate

    index = chain_index(test_chain)
    spot = 650.0
    candidate = put_broken_wing_butterflies(index, config, spot)[0]
    estimate = evaluate(
        candidate.legs, scenarios, CostModel.from_config(config), spot, 1,
        config.float_("risk.es_confidence"),
    )
    # An account small enough that one per-structure budget cannot carry the
    # worst case this structure already computes.
    tiny = context(config, tmp_path, equity=estimate.profile.max_loss * 10.0)
    verdict = MaxLossGate().check(tiny, candidate, estimate, None)
    assert not verdict.passed
    assert verdict.observed == estimate.profile.max_loss
    assert verdict.observed > verdict.threshold


def test_the_leg_count_preference_reports_but_never_blocks(
    test_chain, config, scenarios, tmp_path
):
    """It is a tie-break, not a veto, and the distinction is load-bearing.

    A four-legged structure has to be allowed through on its own merits; the
    preference only decides races at comparable net edge.
    """
    from convex.gates import LegCountPreference

    index = chain_index(test_chain)
    spot = 650.0
    candidate = put_broken_wing_butterflies(index, config, spot)[0]
    estimate = evaluate(
        candidate.legs, scenarios, CostModel.from_config(config), spot, 1,
        config.float_("risk.es_confidence"),
    )
    verdict = LegCountPreference().check(
        context(config, tmp_path), candidate, estimate, None
    )
    assert verdict.blocking is False
    assert verdict.passed
    assert verdict.observed == float(estimate.cost.leg_count)
