"""The liquidity threshold is measured over legs that can actually be traded.

A leg quoted on one side only computes to a relative spread of exactly 2.0,
because with no bid the mid is half the ask. Leaving those in the population
that sets `liquidity.max_relative_spread` drags the median upward and makes the
check more permissive, which is the same defect as setting it from the p90 one
layer down. On the recorded chains this was 34 of 124 legs and 26 of 122,
enough to move the threshold from about 0.104 to about 0.041.
"""

from __future__ import annotations

import pytest

from convex.errors import DataError
from convex.instruments import Right
from scripts.calibrate_costs import tradeable_legs
from tests.conftest import entry


def test_a_leg_with_no_bid_is_left_out_of_the_measured_population():
    chain = [
        entry(760.0, Right.CALL, 1.00, 1.04),
        entry(761.0, Right.CALL, 0.0, 0.05),
        entry(762.0, Right.CALL, 2.00, 2.08),
    ]
    kept = tradeable_legs(chain)
    assert len(kept) == 2
    assert all(row.quote.bid > 0.0 for row in kept)


def test_a_leg_quoted_on_neither_side_is_left_out_too():
    # A bid above a zero ask never reaches this function: Quote refuses it at
    # construction as a crossed quote. What does reach it is a leg with nothing
    # on either side, whose mid is zero and whose relative spread cannot be
    # computed at all.
    assert tradeable_legs([entry(760.0, Right.CALL, 0.0, 0.0)]) == []


def test_the_excluded_legs_are_the_ones_that_would_have_moved_the_threshold():
    # One real leg at about 4% and three unquoted ones. Over the whole
    # population the median is 2.0, a threshold that rejects nothing at all.
    # Over the tradeable legs it is the 4% that was actually there to measure.
    chain = [entry(760.0, Right.CALL, 1.00, 1.04)] + [
        entry(761.0 + offset, Right.CALL, 0.0, 0.05) for offset in range(3)
    ]
    everything = sorted(row.quote.relative_spread for row in chain)
    assert everything[len(everything) // 2] == pytest.approx(2.0)

    kept = tradeable_legs(chain)
    assert len(kept) == 1
    assert kept[0].quote.relative_spread == pytest.approx(0.0392, abs=1e-4)


def test_a_crossed_quote_is_refused_before_it_can_be_measured_at_all():
    with pytest.raises(DataError):
        entry(760.0, Right.CALL, 1.10, 1.00)
