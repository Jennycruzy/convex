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

import numpy as np

from convex.config import Config
from convex.errors import DataError
from convex.features import (
    FeatureSet,
    _strike_widths,
    lagged_results,
    realised_moments,
    time_to_close_years,
)
from convex.archive import ChainSnapshot
from convex.instruments import ChainEntry, OptionContract, Quote, Right

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
    printed: int = 0
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
            "contracts_printed": self.printed,
            "contracts_priced": len(self.entries),
            "dropped_as_inconsistent": max(self.printed - len(self.entries), 0),
            "contracts_requested": self.strikes_requested,
            "coverage": round(self.coverage, 3),
            "implied_vols_solved": len(solved),
            "reconstructed": True,
        }


def arbitrage_free(
    entries: Sequence[ReconstructedEntry], spot: float
) -> list[ReconstructedEntry]:
    """Drop prints that could not have been on the screen together.

    A live chain is one instant of one book, so it satisfies the static
    no-arbitrage conditions by construction. A rebuilt chain does not: each
    strike's print happened at its own moment inside the entry minute, at its
    own side of its own spread, and stitching them into a column produces a
    surface that is merely close to a real one. Priced naively, the gaps show
    up as structures that appear to profit at every expiry price — which the
    sizing code correctly refuses to size, because that is what an arbitrage
    looks like and it is not one.

    So the surface is cleaned to the conditions a real chain would satisfy:

      bounds       a price at least its intrinsic value and below the ceiling
                   the underlying (calls) or the strike (puts) imposes
      monotonicity calls cheapen as the strike rises, puts richen
      convexity    the butterfly around each interior strike is non-negative

    Points that violate are removed, not adjusted. Nudging a print into line
    would invent a trade nobody made; dropping it only narrows the ladder, and
    the coverage figure already reports how much of the ladder survived.
    """
    kept: list[ReconstructedEntry] = []
    for right in (Right.CALL, Right.PUT):
        rows = sorted(
            (e for e in entries if e.contract.right is right),
            key=lambda e: e.contract.strike,
        )
        bounded = []
        for row in rows:
            strike = row.contract.strike
            intrinsic = max(0.0, (spot - strike) if right is Right.CALL else (strike - spot))
            ceiling = spot if right is Right.CALL else strike
            if intrinsic - 1e-6 <= row.entry_price < ceiling:
                bounded.append(row)

        # Monotone in strike: a call is worth less as the strike rises, a put
        # more. The first print of a violating pair is kept and the second
        # dropped, so the ladder stays anchored nearest the money where the
        # tape is thickest.
        monotone: list[ReconstructedEntry] = []
        for row in bounded if right is Right.CALL else list(reversed(bounded)):
            if monotone and row.entry_price > monotone[-1].entry_price + 1e-9:
                continue
            monotone.append(row)
        if right is Right.PUT:
            monotone.reverse()

        # Convex in strike. A strike whose price sits above the chord joining
        # its neighbours makes the butterfly around it negative, so it is the
        # print out of line and it goes. Removing one point can expose another,
        # so this repeats until the side is clean.
        convex = monotone
        changed = True
        while changed and len(convex) >= 3:
            changed = False
            for index in range(1, len(convex) - 1):
                left, middle, right_row = convex[index - 1], convex[index], convex[index + 1]
                span = right_row.contract.strike - left.contract.strike
                if span <= 0:
                    continue
                weight = (right_row.contract.strike - middle.contract.strike) / span
                chord = weight * left.entry_price + (1.0 - weight) * right_row.entry_price
                if middle.entry_price > chord + 1e-6:
                    convex = convex[:index] + convex[index + 1 :]
                    changed = True
                    break
        kept.extend(convex)
    return kept


def _otm(
    entries: Iterable[ReconstructedEntry], spot: float, right: Right
) -> list[ReconstructedEntry]:
    if right is Right.CALL:
        rows = [e for e in entries if e.contract.right is right and e.contract.strike >= spot]
    else:
        rows = [e for e in entries if e.contract.right is right and e.contract.strike <= spot]
    return sorted(rows, key=lambda e: e.contract.strike)


def integrated_variance(
    rows: Sequence[ReconstructedEntry], spot: float, tau: float
) -> float:
    """VIX-style integrated variance from one side, priced off prints.

    The live engine integrates the mid of each out-of-the-money option. There
    is no mid to integrate here, so the print stands in for it. That is a real
    substitution and not a neutral one — a print sits somewhere inside the
    spread rather than at its centre, so this is noisier than the live figure
    and can be biased on a strike that only traded once. It is recorded as a
    reconstructed feature for exactly that reason.
    """
    if tau <= 0.0:
        raise DataError(f"time to expiry must be positive, found {tau}")
    strikes = [row.contract.strike for row in rows]
    widths = _strike_widths(strikes)
    total = 0.0
    for row, width in zip(rows, widths):
        total += (width / row.contract.strike**2) * row.entry_price
    return (2.0 / tau) * total


def smile_slope(rows: Sequence[ReconstructedEntry], spot: float) -> float:
    """Least-squares slope of solved volatility against log-moneyness.

    Only contracts whose print actually solved contribute. A strike that
    printed at parity has no volatility and is left out rather than entered
    as a zero, which would drag the slope toward the money.
    """
    usable = [r for r in rows if r.implied_volatility is not None]
    if len(usable) < 3:
        raise DataError(
            f"smile slope needs three solved volatilities, found {len(usable)}"
        )
    x = np.array([math.log(r.contract.strike / spot) for r in usable])
    y = np.array([r.implied_volatility for r in usable], dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


# What a rebuilt session can supply. The liquidity and open-interest families
# are absent rather than zeroed: the book that produced them is gone, and a
# zero would be a measurement claim nobody made.
RECONSTRUCTED_FEATURES: tuple[str, ...] = (
    "iv_total",
    "iv_up",
    "iv_dn",
    "implied_skew",
    "slope_up",
    "slope_dn",
    "realised_variance",
    "realised_skew",
    "realised_return",
)


def features(
    session: ReconstructedSession,
    entry_at: datetime,
    close_at: datetime,
    prior_returns: Sequence[float],
    family_pnl: dict[str, Sequence[float]],
) -> FeatureSet:
    """The predictor row a rebuilt session can honestly supply."""
    tau = time_to_close_years(entry_at, close_at)
    calls = _otm(session.entries, session.spot_at_entry, Right.CALL)
    puts = _otm(session.entries, session.spot_at_entry, Right.PUT)
    if len(calls) < 2 or len(puts) < 2:
        raise DataError(
            f"{session.session_date}: rebuilt chain has {len(calls)} out-of-the-money "
            f"calls and {len(puts)} puts; integrated variance needs two on each side"
        )

    variance_up = integrated_variance(calls, session.spot_at_entry, tau)
    variance_dn = integrated_variance(puts, session.spot_at_entry, tau)
    values: dict[str, float] = {
        "iv_total": variance_up + variance_dn,
        "iv_up": variance_up,
        "iv_dn": variance_dn,
        "implied_skew": variance_up - variance_dn,
        "slope_up": smile_slope(calls, session.spot_at_entry),
        "slope_dn": smile_slope(puts, session.spot_at_entry),
    }
    values.update(realised_moments(prior_returns))
    for family, history in sorted(family_pnl.items()):
        for name, value in lagged_results(history).items():
            values[f"{family}_{name}"] = value

    return FeatureSet(
        taken_at=entry_at,
        spot=session.spot_at_entry,
        time_to_close_years=tau,
        values=values,
    )


def as_chain_entries(
    session: ReconstructedSession,
    entry_at: datetime,
    relative_spread: float,
) -> list[ChainEntry]:
    """Dress a rebuilt session as a chain, with the spread modelled explicitly.

    The candidate builders, the cost model and the edge calculation read quotes
    and nothing else — no Greeks, no open interest — so a rebuilt session can be
    run through the *same* ranking the live cycle uses rather than a parallel
    one written for the backtest. That matters: a label attached to a candidate
    some other ranking chose is a label for a decision nobody makes.

    The quote is modelled, not observed. The print becomes the mid and the
    spread is the one measured on today's live chain, applied uniformly. Two
    consequences are worth naming. A uniform spread understates the cost of the
    illiquid wings, which are the strikes a broken-wing structure actually
    reaches for. And the depth is unknown, so sizes are zero here and any check
    that reads depth is meaningless on a rebuilt session.

    Greeks and open interest stay None rather than becoming zero, so anything
    downstream that needs them raises instead of quietly reading a fabrication.
    """
    if relative_spread < 0.0:
        raise DataError(f"modelled relative spread must not be negative: {relative_spread}")
    rows: list[ChainEntry] = []
    for entry in session.entries:
        half = 0.5 * relative_spread * entry.entry_price
        bid = entry.entry_price - half
        if bid <= 0.0:
            # A contract whose modelled bid is not positive could not have been
            # sold at any price this model believes in, so it is dropped rather
            # than floored to a tick nobody quoted.
            continue
        rows.append(
            ChainEntry(
                contract=entry.contract,
                quote=Quote(
                    symbol=entry.contract.symbol,
                    bid=bid,
                    ask=entry.entry_price + half,
                    bid_size=0,
                    ask_size=0,
                    timestamp=entry_at,
                ),
                greeks=None,
                open_interest=None,
                volume=entry.volume,
            )
        )
    return rows


def as_snapshot(
    session: ReconstructedSession,
    entry_at: datetime,
    relative_spread: float,
) -> ChainSnapshot:
    """A rebuilt session in the shape the labeller already understands.

    Built in memory and never written to the chain archive: the archive is the
    record of what CONVEX actually saw, and a rebuilt session did not happen to
    it. Keeping the shape lets training.build_samples label these sessions with
    the live ranking rather than a second implementation written for history.
    """
    return ChainSnapshot(
        session_date=session.session_date,
        taken_at=entry_at,
        spot=session.spot_at_entry,
        expiry=session.expiry,
        entries=as_chain_entries(session, entry_at, relative_spread),
    )


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

    # Prints from different moments do not form an arbitrage-free surface on
    # their own, and one that is not arbitrage-free cannot be priced.
    consistent = arbitrage_free(entries, spot_at_entry)

    return ReconstructedSession(
        session_date=session_date,
        expiry=session_date,
        spot_at_entry=spot_at_entry,
        spot_at_close=spot_at_close,
        entries=tuple(consistent),
        strikes_requested=len(contracts),
        printed=len(entries),
    )
