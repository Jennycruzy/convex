"""Enumerating candidate structures from a live chain.

Widths are expressed as a fraction of spot, never as absolute strike points.
The research is SPX at roughly 6,800 with five-point strikes; SPY trades near
650 with one-point strikes, so a width copied across in points would be a
different trade entirely. A fraction of spot survives the translation, which is
why the moneyness band from the paper, 0.98 to 1.02, is the configured input.

Each builder returns structures only, never opinions. Whether a structure is
worth trading is decided later, net of cost, by the ranking and the risk gates.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

from convex.config import Config
from convex.errors import DataError
from convex.instruments import Leg, Right
from convex.structures.base import (
    Candidate,
    ChainIndex,
    Family,
    chain_index,
    nearest,
    strikes_of,
    within_band,
)


def _leg(index: ChainIndex, right: Right, strike: float, ratio: int) -> Leg:
    entry = index.get((right, strike))
    if entry is None:
        raise DataError(f"chain has no {right} at strike {strike}")
    return Leg(entry=entry, ratio=ratio)


def _band(config: Config, spot: float) -> tuple[float, float]:
    return (
        spot * config.float_("candidates.moneyness_low"),
        spot * config.float_("candidates.moneyness_high"),
    )


def _wing_bounds(config: Config, spot: float) -> tuple[float, float]:
    return (
        spot * config.float_("candidates.min_wing_width_pct"),
        spot * config.float_("candidates.max_wing_width_pct"),
    )


def put_broken_wing_butterflies(
    index: ChainIndex, config: Config, spot: float
) -> list[Candidate]:
    """Long one put above, short two at the body, long one further below.

    This is the paper's strongest structure, the put ratio spread, with the
    lower wing bought back so the loss is bounded. The far wing is deliberately
    wider than the near wing: that asymmetry is what leaves the payoff flat and
    positive above every strike when the structure is entered for a credit, so
    a rally costs nothing at all.
    """
    low, high = _band(config, spot)
    min_wing, max_wing = _wing_bounds(config, spot)
    strikes = strikes_of(index, Right.PUT)
    body_strikes = within_band(strikes, low, high)

    candidates: list[Candidate] = []
    for body in body_strikes:
        uppers = [s for s in strikes if min_wing <= s - body <= max_wing]
        lowers = [s for s in strikes if min_wing <= body - s <= max_wing]
        for upper in uppers:
            near_width = upper - body
            for lower in lowers:
                far_width = body - lower
                if far_width <= near_width:
                    # Equal wings are a symmetric butterfly, which this project
                    # does not trade; a narrower far wing inverts the risk.
                    continue
                candidates.append(
                    Candidate(
                        family=Family.PUT_BWB,
                        legs=(
                            _leg(index, Right.PUT, upper, +1),
                            _leg(index, Right.PUT, body, -2),
                            _leg(index, Right.PUT, lower, +1),
                        ),
                        description=(
                            f"put broken-wing butterfly {upper:g}/{body:g}/{lower:g}, "
                            f"near wing {near_width:g} and far wing {far_width:g}"
                        ),
                    )
                )
    return candidates


def call_broken_wing_butterflies(
    index: ChainIndex, config: Config, spot: float
) -> list[Candidate]:
    """The upside mirror of the put structure.

    The research finds call ratio structures weaker than put ratio structures,
    so this family is built but never favoured: it breaks ties last, and it
    trades only when its own classifier says so on its own net edge.
    """
    low, high = _band(config, spot)
    min_wing, max_wing = _wing_bounds(config, spot)
    strikes = strikes_of(index, Right.CALL)
    body_strikes = within_band(strikes, low, high)

    candidates: list[Candidate] = []
    for body in body_strikes:
        lowers = [s for s in strikes if min_wing <= body - s <= max_wing]
        uppers = [s for s in strikes if min_wing <= s - body <= max_wing]
        for lower in lowers:
            near_width = body - lower
            for upper in uppers:
                far_width = upper - body
                if far_width <= near_width:
                    continue
                candidates.append(
                    Candidate(
                        family=Family.CALL_BWB,
                        legs=(
                            _leg(index, Right.CALL, lower, +1),
                            _leg(index, Right.CALL, body, -2),
                            _leg(index, Right.CALL, upper, +1),
                        ),
                        description=(
                            f"call broken-wing butterfly {lower:g}/{body:g}/{upper:g}, "
                            f"near wing {near_width:g} and far wing {far_width:g}"
                        ),
                    )
                )
    return candidates


def straddles(index: ChainIndex, config: Config, spot: float) -> list[Candidate]:
    """Long call and long put at the strike closest to spot."""
    del config
    call_strikes = strikes_of(index, Right.CALL)
    put_strikes = strikes_of(index, Right.PUT)
    shared = sorted(set(call_strikes) & set(put_strikes))
    if not shared:
        raise DataError("chain has no strike with both a call and a put")
    strike = nearest(shared, spot)
    return [
        Candidate(
            family=Family.STRADDLE,
            legs=(
                _leg(index, Right.CALL, strike, +1),
                _leg(index, Right.PUT, strike, +1),
            ),
            description=f"long straddle at {strike:g}",
        )
    ]


def strangles(index: ChainIndex, config: Config, spot: float) -> list[Candidate]:
    """Long an out-of-the-money call and put across the configured widths."""
    min_wing, max_wing = _wing_bounds(config, spot)
    call_strikes = [s for s in strikes_of(index, Right.CALL) if min_wing <= s - spot <= max_wing]
    put_strikes = [s for s in strikes_of(index, Right.PUT) if min_wing <= spot - s <= max_wing]

    return [
        Candidate(
            family=Family.STRANGLE,
            legs=(
                _leg(index, Right.CALL, call_strike, +1),
                _leg(index, Right.PUT, put_strike, +1),
            ),
            description=f"long strangle {put_strike:g}/{call_strike:g}",
        )
        for call_strike in call_strikes
        for put_strike in put_strikes
    ]


def debit_verticals(index: ChainIndex, config: Config, spot: float) -> list[Candidate]:
    """Two-legged directional spreads, bullish in calls and bearish in puts.

    Two legs means two spreads to cross rather than four, which is exactly the
    cost term that turned the research's four-legged structures negative. The
    leg-count preference exists to let this family win a close race.
    """
    low, high = _band(config, spot)
    min_wing, max_wing = _wing_bounds(config, spot)

    candidates: list[Candidate] = []
    for right in (Right.CALL, Right.PUT):
        strikes = within_band(strikes_of(index, right), low, high)
        for lower, upper in combinations(strikes, 2):
            width = upper - lower
            if not min_wing <= width <= max_wing:
                continue
            if right is Right.CALL:
                legs = (_leg(index, right, lower, +1), _leg(index, right, upper, -1))
                label = f"bull call vertical {lower:g}/{upper:g}"
            else:
                legs = (_leg(index, right, upper, +1), _leg(index, right, lower, -1))
                label = f"bear put vertical {upper:g}/{lower:g}"
            candidates.append(
                Candidate(family=Family.DEBIT_VERTICAL, legs=legs, description=label)
            )
    return candidates


_BUILDERS = {
    Family.PUT_BWB: put_broken_wing_butterflies,
    Family.CALL_BWB: call_broken_wing_butterflies,
    Family.STRADDLE: straddles,
    Family.STRANGLE: strangles,
    Family.DEBIT_VERTICAL: debit_verticals,
}


def tradeable_legs(chain: Iterable, floor: float) -> list:
    """The legs worth building a structure out of on an expiration day.

    Two exclusions, both of them about the same thing. A leg quoted on one side
    only cannot be traded in the direction that has no quote, and its mid is
    half the other side by construction, so it prices as though it were cheap
    when in fact nobody is making a market in it.

    A leg worth less than the floor is where an 0DTE book stops being a market.
    On the 1 September SPY chain the 27 legs quoted under a quarter carry a
    median relative spread of 14.29 percent, against 1.33 percent for the 69
    legs above it, and the break-even for this whole strategy sits between 1.0
    and 1.5 percent. Structures built out of those legs are refused later on
    liquidity or on cost, having first crowded the ranking that decides what is
    considered at all. Declining to build them is not a loosened check. It is
    the oldest rule on an expiration day: do not trade the pennies.
    """
    if floor < 0.0:
        raise DataError(f"candidates.min_leg_premium may not be negative, found {floor}")
    kept = []
    for row in chain:
        quote = row.quote
        if quote.bid <= 0.0 or quote.ask <= 0.0:
            continue
        if (quote.bid + quote.ask) / 2.0 < floor:
            continue
        kept.append(row)
    return kept


def build_candidates(
    chain: Iterable, config: Config, spot: float
) -> dict[Family, list[Candidate]]:
    """Every candidate in every enabled family, keyed by family.

    The per-family cap keeps enumeration bounded on a chain with hundreds of
    strikes. It trims the widest structures first, since those carry the most
    cost per unit of exposure, and the trim is reported by the caller.

    The cap is why the premium floor changes more than it looks like it should.
    Removing the pennies does not only remove the structures that contain one,
    it changes which candidates fall inside the narrowest ``cap`` of them, and
    the survivors are better on the measured chain rather than merely fewer.
    That second effect is an interaction with the cap and is worth knowing when
    reading a before and after.
    """
    index = chain_index(tradeable_legs(chain, config.float_("candidates.min_leg_premium")))
    enabled = [Family(name) for name in config.list_("structures.enabled")]
    cap = config.int_("candidates.max_candidates_per_structure")
    if cap <= 0:
        raise DataError(f"candidates.max_candidates_per_structure must be positive, found {cap}")

    built: dict[Family, list[Candidate]] = {}
    for family in enabled:
        builder = _BUILDERS.get(family)
        if builder is None:
            raise DataError(f"structures.enabled names {family!r}, which has no builder")
        candidates = builder(index, config, spot)
        candidates.sort(key=lambda candidate: max(candidate.strikes) - min(candidate.strikes))
        built[family] = candidates[:cap]
    return built
