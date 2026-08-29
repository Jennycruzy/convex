"""Rebuilding past 0DTE sessions from what actually traded.

The archive in convex/archive.py records the chain each live cycle saw, and it
is right that training prefers it: a recorded chain has the two-sided quote the
decision was really made against. But the archive only holds sessions CONVEX
itself has run, and a competition week affords four of them against a burn-in
that wants sixty. Without history the classifier never trains and the fallback
rule decides every session by default.

So this module rebuilds sessions that were never recorded, and is careful to be
honest about what a rebuilt session is and is not.

**What Alpaca will return for a past session.** Trade bars for an expired
contract, if the OCC symbol is constructed rather than looked up — the contract
listing drops expired contracts, but the bars endpoint still answers. Minute
bars for the underlying. That is all.

**What it will not return.** The option book. There is no historical quote
endpoint, only a latest-quote one, so the bid, the ask, the sizes and the
open interest of a past 10:00 are gone. Three things follow and each is a real
limit on what a model built from this can claim:

  1. The liquidity features — half-spread, depth, relative spread, tightness —
     cannot be rebuilt. They are absent from a reconstructed row, not zeroed.
  2. The open-interest exposure proxies cannot be rebuilt either, for the same
     reason, and are absent rather than guessed.
  3. Implied volatility is solved from a *print*, not from a mid. A print is
     what someone paid, which is not the price anyone could have been filled
     at on demand, and thin strikes print rarely.

**Settlement needs no option price.** A 0DTE contract expires into intrinsic
value against the underlying's close, so the terminal value of a structure is
computed from SPY's closing print and the strikes. Only the entry price has to
come from the option tape.

Nothing here is written into the recorded-chain archive, and every row carries
`reconstructed=True` so that a session built this way cannot be mistaken for
one that was observed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterable, Sequence

from convex.config import Config
from convex.errors import DataError
from convex.instruments import OptionContract, Right

# Solver bounds. A 0DTE implied volatility outside this range is not a
# volatility, it is a bad print, and it is dropped rather than clamped.
_VOL_LOW = 1e-4
_VOL_HIGH = 5.0
_VOL_TOLERANCE = 1e-6
_MAX_ITERATIONS = 200


def occ_symbol(underlying: str, expiry: date, right: Right, strike: float) -> str:
    """The OCC 21-character symbol, built rather than looked up.

    get_option_contracts drops contracts once they expire, so a past session
    can only be addressed by constructing its symbols: root, then YYMMDD, then
    C or P, then the strike in thousandths padded to eight digits.
    """
    thousandths = int(round(strike * 1000))
    if abs(thousandths / 1000 - strike) > 1e-9:
        raise DataError(f"strike {strike} is not expressible in OCC thousandths")
    letter = "C" if right is Right.CALL else "P"
    return f"{underlying}{expiry:%y%m%d}{letter}{thousandths:08d}"


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes(
    spot: float, strike: float, right: Right, years: float, rate: float, vol: float
) -> float:
    """European price. Used only to invert a print into an implied volatility.

    SPY options are American, but early exercise on the last few hours of a
    contract's life is worth essentially nothing above intrinsic, and this is
    never used to price a trade — only to read a volatility out of a print.
    """
    if years <= 0.0 or vol <= 0.0:
        return max(0.0, (spot - strike) if right is Right.CALL else (strike - spot))
    root = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / root
    d2 = d1 - root
    discount = math.exp(-rate * years)
    if right is Right.CALL:
        return spot * _normal_cdf(d1) - strike * discount * _normal_cdf(d2)
    return strike * discount * _normal_cdf(-d2) - spot * _normal_cdf(-d1)


def implied_volatility(
    price: float, spot: float, strike: float, right: Right, years: float, rate: float
) -> float | None:
    """Solve a print back into a volatility, or return None if it will not solve.

    Bisection rather than Newton: vega collapses on a 0DTE wing and a Newton
    step there walks off into nonsense. A print below intrinsic or above the
    no-arbitrage ceiling has no implied volatility, and None says so rather
    than a number saying something false.
    """
    if price <= 0.0 or years <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return None
    intrinsic = max(0.0, (spot - strike) if right is Right.CALL else (strike - spot))
    if price < intrinsic - 1e-6:
        return None
    ceiling = spot if right is Right.CALL else strike
    if price >= ceiling:
        return None

    low, high = _VOL_LOW, _VOL_HIGH
    if black_scholes(spot, strike, right, years, rate, high) < price:
        return None

    for _ in range(_MAX_ITERATIONS):
        mid = 0.5 * (low + high)
        value = black_scholes(spot, strike, right, years, rate, mid)
        if abs(value - price) < _VOL_TOLERANCE:
            return mid
        if value < price:
            low = mid
        else:
            high = mid
        if high - low < _VOL_TOLERANCE:
            break

    solved = 0.5 * (low + high)
    # A solution sitting on a bound did not solve: it is a print at parity, or
    # one no volatility reaches. Deep in the money on a thin strike this is the
    # common case, and reporting the floor as though it were a measured 0.01%
    # volatility would poison the skew feature that reads it.
    if solved <= _VOL_LOW * 2.0 or solved >= _VOL_HIGH * 0.99:
        return None
    return solved


@dataclass(frozen=True)
class ReconstructedEntry:
    """One contract of a rebuilt session. No quote, because there is none."""

    contract: OptionContract
    entry_price: float
    implied_volatility: float | None
    trades: int
    volume: int


@dataclass(frozen=True)
class ReconstructedSession:
    """A past 0DTE session rebuilt from prints, and marked as such."""

    session_date: date
    expiry: date
    spot_at_entry: float
    spot_at_close: float
    entries: tuple[ReconstructedEntry, ...]
    strikes_requested: int
    reconstructed: bool = True

    @property
    def coverage(self) -> float:
        """Share of the requested contracts that actually printed at the entry."""
        if self.strikes_requested == 0:
            return 0.0
        return len(self.entries) / self.strikes_requested

    def describe(self) -> dict:
        solved = [e for e in self.entries if e.implied_volatility is not None]
        return {
            "session": self.session_date.isoformat(),
            "spot_at_entry": round(self.spot_at_entry, 2),
            "spot_at_close": round(self.spot_at_close, 2),
            "contracts_priced": len(self.entries),
            "contracts_requested": self.strikes_requested,
            "coverage": round(self.coverage, 3),
            "implied_vols_solved": len(solved),
            "reconstructed": True,
        }


def strike_ladder(spot: float, config: Config) -> list[float]:
    """Whole-dollar strikes inside the configured moneyness band.

    SPY lists in one-dollar increments, which is measured on the live chain
    rather than assumed here: the ladder is built in dollars and any strike the
    exchange did not list simply never prints and drops out.
    """
    low = spot * config.float_("candidates.moneyness_low")
    high = spot * config.float_("candidates.moneyness_high")
    first = math.ceil(low)
    last = math.floor(high)
    if last < first:
        raise DataError(f"moneyness band around {spot} contains no whole-dollar strike")
    return [float(strike) for strike in range(first, last + 1)]


def _bar_price(bar: dict) -> float | None:
    """The volume-weighted price of the entry minute, falling back to its close."""
    for key in ("vw", "c"):
        value = bar.get(key)
        if value is not None and float(value) > 0.0:
            return float(value)
    return None


def build(
    gateway,
    config: Config,
    session_date: date,
    entry_at: datetime,
    close_at: datetime,
    spot_at_entry: float,
    spot_at_close: float,
) -> ReconstructedSession:
    """Rebuild one session's chain at the entry minute.

    The caller supplies the underlying's own prices because those come from the
    stock tape, which is a different request and is worth making once for a run
    of sessions rather than once per session.
    """
    rate = config.float_("reconstruction.risk_free_rate")
    ladder = strike_ladder(spot_at_entry, config)
    contracts: dict[str, OptionContract] = {}
    for strike in ladder:
        for right in (Right.CALL, Right.PUT):
            symbol = occ_symbol(
                config.str_("underlying.symbol"), session_date, right, strike
            )
            contracts[symbol] = OptionContract(
                symbol=symbol,
                underlying=config.str_("underlying.symbol"),
                right=right,
                strike=strike,
                expiry=session_date,
                multiplier=config.int_("underlying.contract_multiplier"),
            )

    # One minute of tape at the entry. A contract that did not trade in that
    # minute has no entry price and is left out rather than filled in.
    bars = gateway.option_bars(
        list(contracts), entry_at, entry_at + timedelta(minutes=1)
    )
    years = max((close_at - entry_at).total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0)

    entries: list[ReconstructedEntry] = []
    for symbol, contract in contracts.items():
        for bar in bars.get(symbol) or []:
            price = _bar_price(bar)
            if price is None:
                continue
            entries.append(
                ReconstructedEntry(
                    contract=contract,
                    entry_price=price,
                    implied_volatility=implied_volatility(
                        price, spot_at_entry, contract.strike, contract.right, years, rate
                    ),
                    trades=int(bar.get("n") or 0),
                    volume=int(bar.get("v") or 0),
                )
            )
            break

    return ReconstructedSession(
        session_date=session_date,
        expiry=session_date,
        spot_at_entry=spot_at_entry,
        spot_at_close=spot_at_close,
        entries=tuple(entries),
        strikes_requested=len(contracts),
    )
