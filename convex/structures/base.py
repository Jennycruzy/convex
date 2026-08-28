"""Candidate structures and the strike bookkeeping they share."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Sequence

from convex.errors import DataError
from convex.instruments import ChainEntry, Leg, Right


class Family(StrEnum):
    """The tradable families, plus the decision not to trade."""

    PUT_BWB = "put_bwb"
    CALL_BWB = "call_bwb"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    DEBIT_VERTICAL = "debit_vertical"
    STAND_DOWN = "stand_down"


@dataclass(frozen=True)
class Candidate:
    """One priceable structure: a family, its legs, and how to say it aloud."""

    family: Family
    legs: tuple[Leg, ...]
    description: str

    def __post_init__(self) -> None:
        if not self.legs:
            raise DataError(f"{self.family}: a candidate needs at least one leg")
        expiries = {leg.contract.expiry for leg in self.legs}
        if len(expiries) != 1:
            raise DataError(f"{self.family}: legs span more than one expiry {sorted(expiries)}")

    @property
    def strikes(self) -> tuple[float, ...]:
        return tuple(leg.contract.strike for leg in self.legs)

    @property
    def short_legs(self) -> tuple[Leg, ...]:
        return tuple(leg for leg in self.legs if leg.is_short)

    def leg_dicts(self) -> list[dict]:
        """Ledger form: enough to reconstruct the position from the receipt."""
        return [
            {
                "symbol": leg.contract.symbol,
                "right": str(leg.contract.right),
                "strike": leg.contract.strike,
                "ratio": leg.ratio,
                "bid": leg.entry.quote.bid,
                "ask": leg.entry.quote.ask,
                "mid": round(leg.entry.quote.mid, 4),
            }
            for leg in self.legs
        ]


ChainIndex = dict[tuple[Right, float], ChainEntry]


def chain_index(chain: Iterable[ChainEntry]) -> ChainIndex:
    """Index a chain snapshot by right and strike for exact strike lookups."""
    index: ChainIndex = {}
    for row in chain:
        key = (row.contract.right, row.contract.strike)
        if key in index:
            raise DataError(f"chain contains two rows for {key[0]} {key[1]}")
        index[key] = row
    if not index:
        raise DataError("chain snapshot is empty")
    return index


def strikes_of(index: ChainIndex, right: Right) -> list[float]:
    return sorted(strike for key_right, strike in index if key_right is right)


def within_band(strikes: Sequence[float], low: float, high: float) -> list[float]:
    return [strike for strike in strikes if low <= strike <= high]


def nearest(strikes: Sequence[float], target: float) -> float:
    if not strikes:
        raise DataError(f"no strikes available near {target}")
    return min(strikes, key=lambda strike: (abs(strike - target), strike))
