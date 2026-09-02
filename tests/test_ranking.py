"""Ranking: net edge decides, and a close race goes to the cheaper execution."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from convex.agent import PricedCandidate, cost_consumed, rank


@dataclass
class _Cost:
    leg_count: int


@dataclass
class _Estimate:
    net_edge: float
    cost: _Cost
    gross_edge: float = 0.0


@dataclass
class _Candidate:
    family: str


def priced(family: str, net_edge: float, legs: int, gross_edge: float = 0.0) -> PricedCandidate:
    return PricedCandidate(_Candidate(family), _Estimate(net_edge, _Cost(legs), gross_edge))


TIE_BREAK = ["put_bwb", "debit_vertical", "straddle", "strangle", "call_bwb"]


def test_a_clearly_better_net_edge_wins_regardless_of_leg_count():
    ordered = rank(
        [priced("debit_vertical", 10.0, 2), priced("put_bwb", 100.0, 3)], TIE_BREAK
    )
    assert ordered[0].candidate.family == "put_bwb"


def test_a_close_race_goes_to_the_structure_with_fewer_legs():
    # Within a tenth of the leader, the two-legged structure wins because each
    # extra leg is another spread to cross.
    ordered = rank(
        [priced("put_bwb", 100.0, 3), priced("debit_vertical", 95.0, 2)], TIE_BREAK
    )
    assert ordered[0].candidate.family == "debit_vertical"


def test_leg_count_ties_fall_back_to_the_configured_family_order():
    ordered = rank(
        [priced("call_bwb", 100.0, 3), priced("put_bwb", 99.0, 3)], TIE_BREAK
    )
    assert ordered[0].candidate.family == "put_bwb"


def test_an_empty_field_ranks_to_nothing():
    assert rank([], TIE_BREAK) == []


def test_a_gross_profit_eaten_entirely_by_cost_is_counted_as_a_cost_refusal():
    # The candidate looked worth taking and stopped being worth taking once the
    # spread was priced. It never reaches a gate, because the ranking demotes it
    # and the walk never gets that far, so this is the only place it can be
    # counted. On the 1 September chain this was 440 of 1,383 priced candidates.
    field = [
        priced("put_bwb", -101.98, 3, gross_edge=7.22),
        priced("call_bwb", -170.52, 3, gross_edge=2.18),
    ]
    assert len(cost_consumed(field)) == 2


def test_a_candidate_that_still_profits_net_is_not_a_cost_refusal():
    assert cost_consumed([priced("straddle", 224.12, 2, gross_edge=227.52)]) == []


def test_a_candidate_with_no_gross_edge_to_lose_is_not_a_cost_refusal():
    # Cost did not take this one away; there was nothing there to take. Counting
    # it would inflate the claim this receipt exists to make.
    assert cost_consumed([priced("call_bwb", -12.0, 3, gross_edge=-8.0)]) == []
