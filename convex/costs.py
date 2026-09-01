"""The cost model.

Law 7: cost is priced at decision time, per leg, before anything is ranked.
The research this project implements found structures with a positive gross
Sharpe and a negative net Sharpe, and the difference was four legs of bid/ask.
So every candidate carries a cost breakdown from the moment it is built, and a
candidate is compared against its rivals net, never gross.

Four charges are levied, all of them per leg and all of them from measured
quotes rather than assumed constants:

  half-spread   crossing from mid to the touch on entry, the dominant term
  slippage      adverse fill beyond the touch, in ticks, calibrated on fills
  fees          per-contract commission and regulatory pass-through
  exit reserve  the same charges again on the legs that must be closed rather
                than left to expire, which is every short leg, because the
                assignment guard closes them before the bell

Leaving a long leg to expire is free, so the reserve is levied on short legs by
default. That choice is configuration, not a constant, because it is a claim
about the world and claims about the world get measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from convex.config import Config
from convex.errors import ConfigError
from convex.instruments import Leg

_EXIT_RESERVE_MODES = {"short", "all", "none"}


@dataclass(frozen=True)
class LegCost:
    """Dollar cost of trading one leg at the requested size."""

    symbol: str
    ratio: int
    contracts: int
    half_spread: float
    slippage: float
    fees: float
    exit_reserve: float

    @property
    def total(self) -> float:
        return self.half_spread + self.slippage + self.fees + self.exit_reserve


@dataclass(frozen=True)
class CostBreakdown:
    """What the dashboard's gross-to-net waterfall is drawn from."""

    contracts: int
    legs: tuple[LegCost, ...]

    @property
    def half_spread(self) -> float:
        return sum(leg.half_spread for leg in self.legs)

    @property
    def slippage(self) -> float:
        return sum(leg.slippage for leg in self.legs)

    @property
    def fees(self) -> float:
        return sum(leg.fees for leg in self.legs)

    @property
    def exit_reserve(self) -> float:
        return sum(leg.exit_reserve for leg in self.legs)

    @property
    def total(self) -> float:
        return sum(leg.total for leg in self.legs)

    @property
    def leg_count(self) -> int:
        """Legs as traded, counting a two-lot short leg once.

        The leg-count preference ranks on this: each distinct leg is another
        spread to cross, which is the term that killed the four-legged
        structures in the research.
        """
        return len(self.legs)

    def as_dict(self) -> dict[str, float]:
        return {
            "half_spread": round(self.half_spread, 4),
            "slippage": round(self.slippage, 4),
            "fees": round(self.fees, 4),
            "exit_reserve": round(self.exit_reserve, 4),
            "total": round(self.total, 4),
        }


@dataclass(frozen=True)
class CostModel:
    """Charges per leg, all of them configured and all of them measurable."""

    slippage_ticks_per_leg: float
    tick_size: float
    per_contract_fee: float
    regulatory_fee_per_contract: float
    exit_reserve_legs: str

    @classmethod
    def from_config(cls, config: Config) -> "CostModel":
        mode = config.str_("costs.exit_reserve_legs")
        if mode not in _EXIT_RESERVE_MODES:
            raise ConfigError(
                f"costs.exit_reserve_legs must be one of {sorted(_EXIT_RESERVE_MODES)}, "
                f"found {mode!r}"
            )
        model = cls(
            slippage_ticks_per_leg=config.float_("costs.slippage_ticks_per_leg"),
            tick_size=config.float_("costs.tick_size"),
            per_contract_fee=config.float_("costs.per_contract_fee"),
            regulatory_fee_per_contract=config.float_("costs.regulatory_fee_per_contract"),
            exit_reserve_legs=mode,
        )
        if model.slippage_ticks_per_leg < 0 or model.tick_size <= 0:
            raise ConfigError("slippage must be non-negative and the tick size positive")
        return model

    def _reserves_exit(self, leg: Leg) -> bool:
        if self.exit_reserve_legs == "all":
            return True
        if self.exit_reserve_legs == "none":
            return False
        return leg.is_short

    def leg_cost(self, leg: Leg, contracts: int) -> LegCost:
        """Cost of one leg of one structure traded ``contracts`` times."""
        if contracts <= 0:
            raise ConfigError(f"cost of {contracts} contracts is not a meaningful quantity")
        units = abs(leg.ratio) * contracts * leg.contract.multiplier
        per_contract_units = abs(leg.ratio) * contracts

        half_spread = leg.entry.quote.half_spread * units
        slippage = self.slippage_ticks_per_leg * self.tick_size * units
        fees = (self.per_contract_fee + self.regulatory_fee_per_contract) * per_contract_units
        reserve = (half_spread + slippage + fees) if self._reserves_exit(leg) else 0.0

        return LegCost(
            symbol=leg.contract.symbol,
            ratio=leg.ratio,
            contracts=contracts,
            half_spread=half_spread,
            slippage=slippage,
            fees=fees,
            exit_reserve=reserve,
        )

    def breakdown(self, legs: Sequence[Leg], contracts: int) -> CostBreakdown:
        return CostBreakdown(
            contracts=contracts,
            legs=tuple(self.leg_cost(leg, contracts) for leg in legs),
        )

    def mid_debit(self, legs: Sequence[Leg]) -> float:
        """Per-share net cash at mid: positive is paid, negative is received."""
        return sum(leg.signed_mid_cost() for leg in legs)

    def executable_debit(self, legs: Sequence[Leg], contracts: int = 1) -> float:
        """Per-share net cash including every entry charge.

        This is the number the risk calculator is given, so the worst case the
        agent sizes against is the worst case it would actually suffer, not the
        one a mid-price fantasy implies.
        """
        breakdown = self.breakdown(legs, contracts)
        entry_cost = breakdown.half_spread + breakdown.slippage + breakdown.fees
        multiplier = legs[0].contract.multiplier
        return self.mid_debit(legs) + entry_cost / (contracts * multiplier)

    def risk_debit(self, legs: Sequence[Leg], contracts: int = 1) -> float:
        """Per-share cash including entry costs and the assignment exit reserve.

        The reserve is a real part of the tail budget for short legs that must
        be bought back before American-style assignment. Omitting it from max
        loss lets the sizer allocate more than the stated risk budget.
        """
        breakdown = self.breakdown(legs, contracts)
        multiplier = legs[0].contract.multiplier
        return self.mid_debit(legs) + breakdown.total / (contracts * multiplier)
