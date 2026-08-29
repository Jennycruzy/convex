"""The checks, fired against a real book.

Law 9: a risk check that has never been observed rejecting something does not
exist. These build real candidates from the real chain, price them net of the
real spread, and assert that the checks reach the verdicts they are supposed to
reach, including the one the whole project turns on, a candidate refused
because its own execution cost consumed its edge.

Nothing here submits an order. The execution path is exercised through the same
dry-run wrapper the live script uses.
"""

from __future__ import annotations

import pytest

from convex.agent import PricedCandidate, rank
from convex.costs import CostModel
from convex.edge import evaluate
from convex.errors import ExecutionError
from convex.scenarios import build as build_scenarios
from convex.structures import build_candidates
from convex.structures.base import Family
from tests.integration.conftest import needs_account


@pytest.fixture(scope="module")
def priced(chain, config, spot):
    """Every candidate the builder produces, priced net of the real spread."""
    from convex.data.alpaca import AlpacaGateway

    connection = AlpacaGateway(config)
    try:
        scenarios = build_scenarios(connection, config)
    finally:
        connection.close()

    cost_model = CostModel.from_config(config)
    confidence = config.float_("risk.es_confidence")
    results = {}
    for family, candidates in build_candidates(chain, config, spot).items():
        results[family] = [
            PricedCandidate(
                candidate,
                evaluate(candidate.legs, scenarios, cost_model, spot, 1, confidence),
            )
            for candidate in candidates
        ]
    return results


@needs_account
def test_the_builder_produces_candidates_from_the_real_chain(priced):
    total = sum(len(rows) for rows in priced.values())
    for family, rows in sorted(priced.items(), key=lambda item: str(item[0])):
        print(f"  {family}: {len(rows)} candidates")
    assert total, "no family produced a single candidate from the live chain"


@needs_account
def test_every_candidate_has_a_computable_bounded_worst_case(priced):
    """Law 5, checked against real prices rather than constructed ones."""
    for family, rows in priced.items():
        for item in rows:
            profile = item.estimate.profile
            assert profile.max_loss > 0.0, f"{family}: a candidate with no downside is a bug"
            assert profile.max_loss < float("inf"), f"{family}: unbounded loss reached pricing"


@needs_account
def test_execution_cost_is_a_real_fraction_of_the_real_edge(priced):
    """The gross-to-net gap, measured on today's book.

    This is the figure the write-up and the video quote, so it is printed and
    kept rather than merely asserted to exist.
    """
    for family, rows in sorted(priced.items(), key=lambda item: str(item[0])):
        if not rows:
            continue
        best = rank(rows, [str(family)])[0]
        estimate = best.estimate
        print(
            f"  {family:<16} gross {estimate.gross_edge:8.2f}  "
            f"cost {estimate.cost.total:7.2f} over {estimate.cost.leg_count} legs  "
            f"net {estimate.net_edge:8.2f}"
        )
        assert estimate.cost.total > 0.0, f"{family}: crossing a real spread costs nothing?"
        assert estimate.cost.half_spread > 0.0


@needs_account
def test_a_candidate_is_refused_because_cost_ate_its_edge(priced, config, spot, gateway):
    """The check the whole project turns on, observed rejecting something.

    If no candidate on the live book fails this, the test says so rather than
    passing quietly. A check that never fires has not been demonstrated, and
    the write-up would be claiming something unobserved.
    """
    from convex.gates import NetOfCostGate

    # The check reads only the estimate, so it needs no session context to be
    # exercised against a real candidate.
    check = NetOfCostGate()
    refused = []
    for family, rows in priced.items():
        for item in rows:
            result = check.check(None, item.candidate, item.estimate, None)
            if not result.passed:
                refused.append((family, item, result))

    total = sum(len(rows) for rows in priced.values())
    print(f"  {len(refused)} of {total} live candidates fail the net-of-cost check")
    assert refused, (
        "not one candidate on the live book was refused for cost. That is possible "
        "on an unusually tight morning, but it means this check has not been "
        "observed firing today and the claim should not be made until it has."
    )

    family, item, result = refused[0]
    print(f"  example: {family} gross {item.estimate.gross_edge:.2f} "
          f"less cost {item.estimate.cost.total:.2f} = {item.estimate.net_edge:.2f}")
    assert item.estimate.net_edge <= 0.0


@needs_account
def test_the_ranking_prefers_fewer_legs_when_the_net_edge_is_close(priced, config):
    """Each leg is another half-spread, so a tie goes to the cheaper structure."""
    everything = [item for rows in priced.values() for item in rows]
    if len(everything) < 2:
        pytest.skip("too few live candidates to rank")
    tie_break = [str(name) for name in config.list_("structures.tie_break_order")]
    ordered = rank(everything, tie_break)
    leader = ordered[0]
    band = abs(ordered[0].estimate.net_edge) * 0.10
    near = [item for item in ordered if leader.estimate.net_edge - item.estimate.net_edge <= band]
    assert leader.estimate.cost.leg_count == min(item.estimate.cost.leg_count for item in near)


@needs_account
def test_the_executor_refuses_a_structure_it_cannot_send(gateway, priced):
    """The order path's own guards, checked without sending anything."""
    everything = [item for rows in priced.values() for item in rows]
    if not everything:
        pytest.skip("no live candidates")
    legs = list(everything[0].candidate.legs)

    with pytest.raises(ExecutionError, match="refusing to submit an order for 0"):
        gateway.submit_structure(legs, 0, 1.0, "convex-test-zero")
    with pytest.raises(ExecutionError, match="no legs"):
        gateway.submit_structure([], 1, 1.0, "convex-test-empty")
    with pytest.raises(ExecutionError, match="at most four legs"):
        gateway.submit_structure(legs * 3, 1, 1.0, "convex-test-toomany")


@needs_account
def test_closing_guards_refuse_an_impossible_close(gateway, chain):
    symbol = chain[0].contract.symbol
    with pytest.raises(ExecutionError, match="zero contracts"):
        gateway.close_leg(symbol, 0, 1.0, "convex-test-close-zero")
    with pytest.raises(ExecutionError, match="limit of"):
        gateway.close_leg(symbol, 1, 0.0, "convex-test-close-free")
