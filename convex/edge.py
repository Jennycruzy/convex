"""Gross edge, net edge, and the tail.

This is where Law 7 and Law 8 meet. One pass over the scenario set produces
every number the ranking and the sizing need:

  gross edge   mean payoff across scenarios at mid prices
  cost         the measured cost model applied to the same legs
  net edge     gross less cost, which is the only figure a candidate is ranked
               on and the only figure the net-of-cost hurdle reads
  ES(1%)       mean of the worst one percent of net outcomes, the tail the
               sizer budgets against instead of the mean
  win rate     share of scenarios with a positive net outcome

The gross-to-net waterfall on the dashboard is this object, drawn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import sqrt
from statistics import NormalDist

import numpy as np

from convex.costs import CostBreakdown, CostModel
from convex.errors import DataError
from convex.instruments import Leg
from convex.payoff import RiskProfile, risk_profile
from convex.scenarios import ScenarioSet


@dataclass(frozen=True)
class EdgeEstimate:
    """Everything known about one candidate at one size."""

    contracts: int
    gross_edge: float
    cost: CostBreakdown
    net_outcomes: np.ndarray
    expected_shortfall: float
    win_rate: float
    profile: RiskProfile
    spot: float

    @property
    def net_edge(self) -> float:
        return self.gross_edge - self.cost.total

    def net_edge_lower_bound(self, confidence: float) -> float:
        """One-sided lower confidence bound for the scenario mean net P&L.

        The point estimate alone is not enough to spend capital on. This uses
        the empirical standard error of the same net scenario outcomes used by
        the tail calculation, and a one-sided normal critical value. It is a
        deliberately conservative admission test, not a claim that historical
        scenarios are independent or a guarantee of future P&L.
        """
        if not 0.5 < confidence < 1.0:
            raise DataError(
                f"net-edge confidence must lie strictly between 0.5 and 1, found {confidence}"
            )
        if self.net_outcomes.size < 2:
            raise DataError("at least two net outcomes are required for a confidence bound")
        standard_error = float(self.net_outcomes.std(ddof=1)) / sqrt(self.net_outcomes.size)
        return self.net_edge - NormalDist().inv_cdf(confidence) * standard_error

    @property
    def cost_share_of_gross(self) -> float:
        """How much of the gross edge the execution takes.

        Reported for the waterfall. A value above one is the case the research
        describes: a structure with real gross alpha and no net alpha at all.
        """
        if self.gross_edge <= 0.0:
            raise DataError(
                "cost share is only meaningful against a positive gross edge; "
                f"this candidate's gross edge is {self.gross_edge:.2f}"
            )
        return self.cost.total / self.gross_edge

    @property
    def es_pct_of_underlying(self) -> float:
        """ES(1%) as a fraction of notional, the units the research reports."""
        notional = self.spot * 100.0 * self.contracts
        if notional <= 0.0:
            raise DataError("cannot express the tail against non-positive notional")
        return self.expected_shortfall / notional

    def waterfall(self) -> dict[str, float]:
        """The bars of the gross-to-net chart, in the order they are drawn."""
        return {
            "gross_edge": round(self.gross_edge, 2),
            "half_spread": -round(self.cost.half_spread, 2),
            "slippage": -round(self.cost.slippage, 2),
            "fees": -round(self.cost.fees, 2),
            "exit_reserve": -round(self.cost.exit_reserve, 2),
            "net_edge": round(self.net_edge, 2),
        }


def at_limit(
    estimate: EdgeEstimate, legs: Sequence[Leg], cost_model: CostModel, limit_price: float,
    es_confidence: float,
) -> EdgeEstimate:
    """Reprice an estimate at an actual limit rather than its modeled touch.

    A worse limit is a cash concession, not free liquidity: it shifts every
    scenario payoff, increases max loss, and is re-tested by the normal gates.
    """
    modeled = cost_model.executable_debit(legs, estimate.contracts)
    concession = limit_price - modeled
    dollars = concession * legs[0].contract.multiplier * estimate.contracts
    outcomes = estimate.net_outcomes - dollars
    return replace(
        estimate,
        gross_edge=estimate.gross_edge - dollars,
        net_outcomes=outcomes,
        expected_shortfall=expected_shortfall(outcomes, es_confidence),
        win_rate=float((outcomes > 0.0).mean()),
        profile=risk_profile(legs, estimate.profile.net_entry_debit + concession),
    )


def expected_shortfall(outcomes: np.ndarray, confidence: float) -> float:
    """Mean loss in the worst ``confidence`` tail, as a positive number.

    At least one scenario always enters the tail, so a small scenario set
    degrades into the single worst outcome rather than into a silent zero.
    """
    if not 0.0 < confidence < 1.0:
        raise DataError(f"ES confidence must lie strictly between 0 and 1, found {confidence}")
    if outcomes.size == 0:
        raise DataError("cannot compute a tail from an empty outcome set")
    count = max(1, int(np.floor(outcomes.size * confidence)))
    worst = np.sort(outcomes)[:count]
    return float(-worst.mean())


def evaluate(
    legs: Sequence[Leg],
    scenarios: ScenarioSet,
    cost_model: CostModel,
    spot: float,
    contracts: int,
    es_confidence: float,
) -> EdgeEstimate:
    """Price one candidate against the scenario set, net of measured cost."""
    if contracts <= 0:
        raise DataError(f"cannot evaluate a candidate at {contracts} contracts")

    mid_debit = cost_model.mid_debit(legs)
    cost = cost_model.breakdown(legs, contracts)
    multiplier = legs[0].contract.multiplier

    terminal = scenarios.prices(spot)
    intrinsic = np.zeros_like(terminal)
    for leg in legs:
        contract = leg.contract
        if contract.right.value == "call":
            leg_value = np.maximum(terminal - contract.strike, 0.0)
        else:
            leg_value = np.maximum(contract.strike - terminal, 0.0)
        intrinsic += leg.ratio * leg_value

    gross_outcomes = (intrinsic - mid_debit) * multiplier * contracts
    net_outcomes = gross_outcomes - cost.total

    return EdgeEstimate(
        contracts=contracts,
        gross_edge=float(gross_outcomes.mean()),
        cost=cost,
        net_outcomes=net_outcomes,
        expected_shortfall=expected_shortfall(net_outcomes, es_confidence),
        win_rate=float((net_outcomes > 0.0).mean()),
        # Size against the same all-in friction used in net outcomes. In
        # particular, short-leg exit reserves cannot disappear from max loss.
        profile=risk_profile(legs, cost_model.risk_debit(legs, contracts)),
        spot=spot,
    )
