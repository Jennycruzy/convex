"""The candidate builders, checked against the shapes they claim to build."""

from __future__ import annotations

import pytest

from convex.config import load
from convex.instruments import Right
from convex.payoff import payoff_at, upside_slope
from convex.structures import Family, build_candidates
from convex.structures.base import chain_index
from tests.conftest import build_test_chain
from convex.structures.builders import (
    call_broken_wing_butterflies,
    debit_verticals,
    put_broken_wing_butterflies,
    straddles,
)


@pytest.fixture
def config():
    return load()


def test_put_broken_wings_are_never_symmetric(test_chain, config):
    built = put_broken_wing_butterflies(chain_index(test_chain), config, 650.0)
    assert built
    for candidate in built:
        upper, body, lower = candidate.strikes
        assert upper > body > lower
        assert (body - lower) > (upper - body)  # the far wing is always wider


def test_put_broken_wing_has_a_flat_upside_and_a_floor(test_chain, config):
    built = put_broken_wing_butterflies(chain_index(test_chain), config, 650.0)
    candidate = built[0]
    assert upside_slope(candidate.legs) == 0  # no calls, so the upside is flat
    high = payoff_at(candidate.legs, 0.10, 900.0)
    higher = payoff_at(candidate.legs, 0.10, 5_000.0)
    assert high == pytest.approx(higher)
    floor = payoff_at(candidate.legs, 0.10, 0.0)
    assert floor == pytest.approx(payoff_at(candidate.legs, 0.10, min(candidate.strikes) - 1))


def test_call_broken_wings_cover_their_short_calls(test_chain, config):
    built = call_broken_wing_butterflies(chain_index(test_chain), config, 650.0)
    assert built
    for candidate in built:
        assert upside_slope(candidate.legs) == 0


def test_verticals_have_two_legs_and_bounded_width(test_chain, config):
    built = debit_verticals(chain_index(test_chain), config, 650.0)
    assert built
    widths = {max(c.strikes) - min(c.strikes) for c in built}
    assert min(widths) >= 650.0 * config.float_("candidates.min_wing_width_pct")
    assert max(widths) <= 650.0 * config.float_("candidates.max_wing_width_pct")
    for candidate in built:
        assert len(candidate.legs) == 2


def test_straddle_sits_on_the_strike_closest_to_spot(test_chain, config):
    built = straddles(chain_index(test_chain), config, 650.4)
    assert built[0].strikes == (650.0, 650.0)
    assert {leg.contract.right for leg in built[0].legs} == {Right.CALL, Right.PUT}


def test_every_enabled_family_produces_candidates_within_the_cap(test_chain, config):
    built = build_candidates(test_chain, config, 650.0)
    assert set(built) == {Family(name) for name in config.list_("structures.enabled")}
    cap = config.int_("candidates.max_candidates_per_structure")
    for family, candidates in built.items():
        assert candidates, f"{family} produced nothing"
        assert len(candidates) <= cap


def test_the_premium_floor_keeps_the_pennies_out_of_every_structure(test_chain, config):
    """Law 7 reaches candidate construction, not only the checks after it.

    A leg quoted at a few cents on an expiration day carries a relative spread
    an order of magnitude past the strategy's break-even. Structures built out
    of one are refused downstream, having first taken up room in the ranking.
    """
    floor = config.float_("candidates.min_leg_premium")
    assert floor > 0.0

    built = build_candidates(test_chain, config, 650.0)

    mids = {
        (leg.contract.right, leg.contract.strike)
        for candidates in built.values()
        for candidate in candidates
        for leg in candidate.legs
    }
    by_key = {(row.contract.right, row.contract.strike): row for row in test_chain}
    for key in mids:
        quote = by_key[key].quote
        assert (quote.bid + quote.ask) / 2.0 >= floor, f"{key} is below the floor"


def test_a_leg_quoted_on_one_side_only_is_never_built_on(config):
    from convex.structures.builders import tradeable_legs

    chain = build_test_chain()
    one_sided = [row for row in chain if row.contract.strike == 650.0]
    assert one_sided
    for row in one_sided:
        object.__setattr__(row.quote, "bid", 0.0)

    kept = tradeable_legs(chain, 0.0)

    assert all(row.quote.bid > 0.0 and row.quote.ask > 0.0 for row in kept)
    assert len(kept) == len(chain) - len(one_sided)
