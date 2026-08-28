"""Market objects: contracts, quotes, Greeks, and one leg of a structure.

Law 3 is enforced at the boundary rather than at the point of use. A quote that
cannot be priced, a chain row missing the Greeks a caller needs, or a snapshot
older than the staleness budget raises here, so that no downstream calculation
ever runs on a value that was quietly substituted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import StrEnum

from convex.errors import DataError, StaleDataError


class Right(StrEnum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class OptionContract:
    """One listed option, as Alpaca describes it."""

    symbol: str
    underlying: str
    right: Right
    strike: float
    expiry: date
    multiplier: int

    def intrinsic(self, spot: float) -> float:
        """Per-share intrinsic value at expiry for the given underlying price."""
        if self.right is Right.CALL:
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)

    def is_itm(self, spot: float) -> bool:
        return self.intrinsic(spot) > 0.0


@dataclass(frozen=True)
class Greeks:
    """Greeks and implied volatility as published by Alpaca for one contract."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_volatility: float


@dataclass(frozen=True)
class Quote:
    """A two-sided quote. Every derived price raises rather than defaulting."""

    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.bid < 0.0 or self.ask < 0.0:
            raise DataError(f"{self.symbol}: negative quote bid={self.bid} ask={self.ask}")
        if self.ask < self.bid:
            raise DataError(f"{self.symbol}: crossed quote bid={self.bid} ask={self.ask}")
        if self.timestamp.tzinfo is None:
            raise DataError(f"{self.symbol}: quote timestamp is not timezone aware")

    @property
    def mid(self) -> float:
        mid = (self.bid + self.ask) / 2.0
        if mid <= 0.0:
            raise DataError(f"{self.symbol}: quote has no priceable mid (bid={self.bid} ask={self.ask})")
        return mid

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def half_spread(self) -> float:
        """The per-share cost of crossing from mid to the touch."""
        return self.spread / 2.0

    @property
    def relative_spread(self) -> float:
        """Spread as a fraction of mid. The liquidity check reads this."""
        return self.spread / self.mid

    def age_seconds(self, now: datetime | None = None) -> float:
        reference = now if now is not None else datetime.now(timezone.utc)
        return (reference - self.timestamp).total_seconds()

    def require_fresh(self, max_age_seconds: float, now: datetime | None = None) -> "Quote":
        """Return self, or raise if this quote is outside the staleness budget."""
        age = self.age_seconds(now)
        if age > max_age_seconds:
            raise StaleDataError(
                f"{self.symbol}: quote is {age:.1f}s old, budget is {max_age_seconds:.1f}s"
            )
        return self


@dataclass(frozen=True)
class ChainEntry:
    """One row of a 0DTE chain snapshot: contract, quote, Greeks, open interest."""

    contract: OptionContract
    quote: Quote
    greeks: Greeks | None
    open_interest: int | None
    volume: int | None

    def require_greeks(self) -> Greeks:
        """Greeks or a loud failure. A missing gamma is not a zero gamma."""
        if self.greeks is None:
            raise DataError(f"{self.contract.symbol}: Alpaca returned no Greeks for this contract")
        return self.greeks

    def require_open_interest(self) -> int:
        if self.open_interest is None:
            raise DataError(f"{self.contract.symbol}: no open interest available")
        return self.open_interest


@dataclass(frozen=True)
class Leg:
    """One leg of a structure, in units of one structure.

    ``ratio`` is signed: positive is long, negative is short. A 1x2 put ratio
    carries ratios of +1 and -2, and the protective wing that turns it into a
    broken-wing butterfly carries +1.
    """

    entry: ChainEntry
    ratio: int

    def __post_init__(self) -> None:
        if self.ratio == 0:
            raise DataError(f"{self.entry.contract.symbol}: a leg with ratio zero is not a leg")

    @property
    def contract(self) -> OptionContract:
        return self.entry.contract

    @property
    def is_short(self) -> bool:
        return self.ratio < 0

    def entry_price(self) -> float:
        """Per-share mid price of this leg's contract."""
        return self.entry.quote.mid

    def signed_mid_cost(self) -> float:
        """Per-share cash effect at mid: positive is paid, negative is received."""
        return self.ratio * self.entry.quote.mid
