"""Ranking: net edge decides, and a close race goes to the cheaper execution."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from convex.agent import PricedCandidate, rank


@dataclass
class _Cost:
    leg_count: int


@dataclass
class _Estimate:
    net_edge: float
    cost: _Cost


@dataclass
class _Candidate:
    family: str


def priced(family: str, net_edge: float, legs: int) -> PricedCandidate:
    return PricedCandidate(_Candidate(family), _Estimate(net_edge, _Cost(legs)))


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
