"""Expiry payoff, maximum loss, and breakevens.

Law 5 lives here. Every structure the agent can build is priced through
max_loss(), and max_loss() raises rather than returning a number it cannot
justify. That is what makes the agent structurally incapable of submitting an
order whose worst case it does not know: there is no other path to a size.

The payoff of any combination of same-expiry options is piecewise linear in the
underlying, with kinks only at the strikes. The minimum of a piecewise linear
function is attained at a vertex whenever the outer segments do not slope down,
so evaluating the payoff at zero and at every strike is exact rather than a
numerical search, provided the far-upside slope is non-negative. A negative
far-upside slope is precisely an uncovered short call, and that is refused.

These are expiry payoffs. SPY options are American, so a short leg can be
assigned early; the assignment guard in the position manager is what keeps the
realised worst case inside the number computed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from convex.errors import UndefinedRiskError
from convex.instruments import Leg, Right


def _multiplier(legs: Sequence[Leg]) -> int:
    if not legs:
        raise UndefinedRiskError("a structure with no legs has no computable risk")
    multipliers = {leg.contract.multiplier for leg in legs}
    if len(multipliers) != 1:
        raise UndefinedRiskError(f"legs carry mixed contract multipliers: {sorted(multipliers)}")
    multiplier = multipliers.pop()
    if multiplier <= 0:
        raise UndefinedRiskError(f"contract multiplier must be positive, found {multiplier}")
    return multiplier


def intrinsic_value(legs: Sequence[Leg], spot: float) -> float:
    """Per-share value of the leg combination at expiry, before entry cost."""
    return sum(leg.ratio * leg.contract.intrinsic(spot) for leg in legs)


def payoff_at(legs: Sequence[Leg], net_entry_debit: float, spot: float) -> float:
    """Profit or loss in dollars for one structure at an expiry price.

    ``net_entry_debit`` is per share and signed the way cash moves: positive
    when the structure was paid for, negative when it was entered for a credit.
    Pass the cost-inclusive debit to get a cost-inclusive payoff.
    """
    multiplier = _multiplier(legs)
    return multiplier * (intrinsic_value(legs, spot) - net_entry_debit)


def upside_slope(legs: Sequence[Leg]) -> int:
    """Net long call units, which is the payoff slope above the highest strike."""
    return sum(leg.ratio for leg in legs if leg.contract.right is Right.CALL)


def vertices(legs: Sequence[Leg]) -> list[float]:
    """The prices at which the payoff can kink, plus the zero boundary."""
    return sorted({0.0} | {leg.contract.strike for leg in legs})


@dataclass(frozen=True)
class RiskProfile:
    """The bounded worst case of a structure, and where it occurs."""

    max_loss: float
    max_loss_price: float
    max_profit: float
    max_profit_price: float
    net_entry_debit: float
    breakevens: tuple[float, ...]

    @property
    def is_credit(self) -> bool:
        return self.net_entry_debit < 0.0


def max_loss(legs: Sequence[Leg], net_entry_debit: float) -> tuple[float, float]:
    """Return (worst case loss in dollars, the price at which it occurs).

    The loss is reported as a positive number. Raises UndefinedRiskError when
    the far-upside payoff slopes down, which is an uncovered short call, and
    when the structure shows a profit at every expiry price, which is an
    apparent arbitrage and in practice means the quotes are wrong.
    """
    slope = upside_slope(legs)
    if slope < 0:
        raise UndefinedRiskError(
            f"payoff slopes down above the highest strike by {slope} call units: "
            "loss is unbounded to the upside and the structure is refused"
        )

    knots = vertices(legs)
    worst_payoff = min(payoff_at(legs, net_entry_debit, price) for price in knots)
    # The payoff is flat across a tie, so report the highest price attaining the
    # worst case: for a put structure that is the strike the loss plateau starts
    # at, which is more informative than the zero boundary.
    worst_price = max(
        price for price in knots if payoff_at(legs, net_entry_debit, price) == worst_payoff
    )
    if worst_payoff >= 0.0:
        raise UndefinedRiskError(
            f"structure shows a profit of {worst_payoff:.2f} at every expiry price; "
            "refusing to size an apparent arbitrage, the quotes should be re-read"
        )
    return -worst_payoff, worst_price


def breakevens(legs: Sequence[Leg], net_entry_debit: float) -> tuple[float, ...]:
    """Prices at which the expiry payoff crosses zero, exactly.

    Each segment between kinks is linear, so a sign change is solved rather
    than searched. The segment above the highest strike is included when its
    slope can carry the payoff across zero.
    """
    knots = vertices(legs)
    slope = upside_slope(legs)
    if slope > 0:
        span = max(knots[-1], 1.0)
        knots = knots + [knots[-1] + span]

    crossings: list[float] = []
    for left, right in zip(knots, knots[1:]):
        left_payoff = payoff_at(legs, net_entry_debit, left)
        right_payoff = payoff_at(legs, net_entry_debit, right)
        if left_payoff == 0.0:
            crossings.append(left)
        if (left_payoff < 0.0) != (right_payoff < 0.0) and left_payoff != 0.0:
            weight = left_payoff / (left_payoff - right_payoff)
            crossings.append(left + weight * (right - left))
    if payoff_at(legs, net_entry_debit, knots[-1]) == 0.0:
        crossings.append(knots[-1])
    return tuple(sorted(set(round(price, 6) for price in crossings)))


def risk_profile(legs: Sequence[Leg], net_entry_debit: float) -> RiskProfile:
    """The full bounded risk description used by the sizer and the dashboard."""
    loss, loss_price = max_loss(legs, net_entry_debit)
    knots = vertices(legs)
    slope = upside_slope(legs)
    if slope > 0:
        # Unbounded profit above the highest strike; report the payoff one full
        # span above it so the figure is finite and its provenance is obvious.
        knots = knots + [knots[-1] * 2.0]
    best_price = max(knots, key=lambda price: payoff_at(legs, net_entry_debit, price))
    return RiskProfile(
        max_loss=loss,
        max_loss_price=loss_price,
        max_profit=payoff_at(legs, net_entry_debit, best_price),
        max_profit_price=best_price,
        net_entry_debit=net_entry_debit,
        breakevens=breakevens(legs, net_entry_debit),
    )


def payoff_curve(
    legs: Sequence[Leg], net_entry_debit: float, low: float, high: float, points: int = 241
) -> list[tuple[float, float]]:
    """Sampled payoff for the dashboard's diagrams, kinks included exactly."""
    if points < 2:
        raise UndefinedRiskError("a payoff curve needs at least two points")
    if not high > low:
        raise UndefinedRiskError(f"payoff curve bounds are inverted: low={low} high={high}")
    step = (high - low) / (points - 1)
    grid = {low + step * index for index in range(points)}
    grid |= {strike for strike in vertices(legs) if low <= strike <= high}
    return [(price, payoff_at(legs, net_entry_debit, price)) for price in sorted(grid)]
