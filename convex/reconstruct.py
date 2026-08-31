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
contract, if the OCC symbol is constructed rather than looked up. The contract
listing drops expired contracts, but the bars endpoint still answers. Minute
bars for the underlying. That is all.

**What it will not return.** The option book. There is no historical quote
endpoint, only a latest-quote one, so the bid, the ask, the sizes and the
open interest of a past 10:00 are gone. Three things follow and each is a real
limit on what a model built from this can claim:

  1. The liquidity features, being half-spread, depth, relative spread and
     tightness, cannot be rebuilt. They are absent from a reconstructed row, not zeroed.
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
from zoneinfo import ZoneInfo
from typing import Iterable, Sequence

import numpy as np

from convex.config import Config
from convex.errors import DataError
from convex.features import (
    FeatureSet,
    _strike_widths,
    lagged_results,
    realised_moments,
    tape_features,
    time_to_close_years,
)
from convex.archive import ChainSnapshot
from convex.instruments import ChainEntry, OptionContract, Quote, Right

# The solver lives in its own module because the live 0DTE path needs the same
# one: Alpaca publishes no volatility on expiration day. Re-exported here so a
# reader of the rebuild finds it where the rebuild uses it.
from convex.volatility import black_scholes, implied_volatility  # noqa: F401


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
    up as structures that appear to profit at every expiry price, which the
    sizing code correctly refuses to size, because that is what an arbitrage
    looks like and it is not one.

    So the surface is cleaned to the conditions a real chain would satisfy:

      bounds       a price at least its intrinsic value and below the ceiling
                   the underlying (calls) or the strike (puts) imposes
      monotonicity calls cheapen as the strike rises, puts richen
      convexity    the butterfly around each interior strike is non-negative
      verticals    a spread costs no more than the distance between its strikes

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

        # Convex in strike, and no vertical that costs more than it can pay.
        #
        # Convexity alone was not enough, and the sweep is what found it.
        # Convexity bounds the butterfly weighted by the two wing widths, but a
        # broken-wing butterfly is weighted 1/-2/1 instead, and those are the
        # same weights only when the wings are equal, which is the one case
        # this project never trades. So a surface can be perfectly convex and
        # still price a broken wing at a credit larger than the gap between its
        # wings, which pays at every expiry price and is an arbitrage. The
        # missing condition is the vertical: a spread cannot cost more than the
        # distance between its strikes, because that is the most it can ever
        # pay. Convexity plus that bound rules the credit out.
        #
        # Under convexity the steepest vertical is the outermost pair, deepest
        # in the money, where a strike prints rarely and a stale print sits
        # furthest from where the book was. That is the end a violation trims.
        cleaned = monotone
        changed = True
        while changed and len(cleaned) >= 2:
            changed = False

            for index in range(1, len(cleaned) - 1):
                left, middle, right_row = cleaned[index - 1], cleaned[index], cleaned[index + 1]
                span = right_row.contract.strike - left.contract.strike
                if span <= 0:
                    continue
                weight = (right_row.contract.strike - middle.contract.strike) / span
                chord = weight * left.entry_price + (1.0 - weight) * right_row.entry_price
                if middle.entry_price > chord + 1e-6:
                    cleaned = cleaned[:index] + cleaned[index + 1 :]
                    changed = True
                    break
            if changed:
                continue

            # Calls steepen towards the low strikes and puts towards the high
            # ones, so each side is scanned from its own deep-in-the-money end
            # and the outer print of the offending pair is the one dropped.
            if right is Right.CALL:
                # index is the lower strike of the pair, and the outer print.
                pairs = [(index, index, index + 1) for index in range(len(cleaned) - 1)]
            else:
                # index is the higher strike of the pair, and the outer print.
                pairs = [(index, index - 1, index) for index in range(len(cleaned) - 1, 0, -1)]
            for drop, low, high in pairs:
                lower, upper = cleaned[low], cleaned[high]
                width = upper.contract.strike - lower.contract.strike
                if width <= 0:
                    continue
                cost = (
                    lower.entry_price - upper.entry_price
                    if right is Right.CALL
                    else upper.entry_price - lower.entry_price
                )
                if cost > width + 1e-6:
                    cleaned = cleaned[:drop] + cleaned[drop + 1 :]
                    changed = True
                    break
        kept.extend(cleaned)
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
    substitution and not a neutral one. A print sits somewhere inside the
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
    "tape_volume",
    "tape_breadth",
    "tape_concentration",
    "tape_put_share",
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
    values.update(
        tape_features([(e.contract.right, e.volume) for e in session.entries])
    )
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
    and nothing else, with no Greeks and no open interest, so a rebuilt session can be
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


def _stamp_of(bar: dict, symbol: str) -> datetime:
    """A bar's timestamp, required rather than assumed."""
    value = bar.get("t")
    if value is None:
        raise DataError(f"{symbol}: minute bar has no 't' (keys: {sorted(bar)})")
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _count(bar: dict, key: str, symbol: str) -> int:
    """A bar's trade count or volume, required rather than defaulted.

    Law 3. These arrive on every bar Alpaca returns, so an absent one means the
    shape changed underneath this code, and a zero there would read as a
    contract that printed without trading.
    """
    value = bar.get(key)
    if value is None:
        raise DataError(f"{symbol}: minute bar has no {key!r} (keys: {sorted(bar)})")
    return int(value)


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
    open_at: datetime | None = None,
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

    # From the opening bell to the entry, not the entry minute alone. The price
    # comes from the entry minute, but the volume has to be session-to-date,
    # because that is what the live snapshot reports and a feature that means
    # one thing in training and another at 10:00 is worse than no feature.
    started = open_at or entry_at - timedelta(minutes=30)
    bars = gateway.option_bars(
        list(contracts), started, entry_at + timedelta(minutes=1)
    )
    years = max((close_at - entry_at).total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0)

    entries: list[ReconstructedEntry] = []
    for symbol, contract in contracts.items():
        session_bars = bars.get(symbol) or []
        if not session_bars:
            continue
        # Everything that printed at or before the entry, so the volume is the
        # session's and the price is the entry minute's.
        upto = [b for b in session_bars if _stamp_of(b, symbol) <= entry_at]
        if not upto:
            continue
        last = upto[-1]
        price = _bar_price(last)
        if price is None:
            continue
        entries.append(
            ReconstructedEntry(
                contract=contract,
                entry_price=price,
                implied_volatility=implied_volatility(
                    price, spot_at_entry, contract.strike, contract.right, years, rate
                ),
                trades=sum(_count(b, "n", symbol) for b in upto),
                volume=sum(_count(b, "v", symbol) for b in upto),
            )
        )

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


@dataclass(frozen=True)
class RebuiltWindow:
    """Every session a window could rebuild, ready to label or replay."""

    snapshots: list[ChainSnapshot]
    settlements: dict[date, float]
    sessions: dict[date, ReconstructedSession]
    coverages: list[float]
    dropped_thin: int
    modelled_relative_spread: float

    def describe(self) -> dict:
        import statistics

        return {
            "reconstructed": True,
            "sessions": len(self.snapshots),
            "median_coverage": (
                round(statistics.median(self.coverages), 3) if self.coverages else 0.0
            ),
            "dropped_thin": self.dropped_thin,
            "modelled_relative_spread": self.modelled_relative_spread,
        }

    def feature_builder(self):
        """The predictor builder these sessions must be labelled with.

        Handed to training.build_samples in place of the live feature engine,
        which reads Greeks and a book that a rebuilt session does not have.
        """

        def build_features(entries, spot, taken_at, close_at, prior_returns, family_pnl):
            return features(
                self.sessions[taken_at.date()], taken_at, close_at, prior_returns, family_pnl
            )

        return build_features


# How far behind live the complimentary option feed runs. Asking it for
# anything newer is refused with "OPRA agreement is not signed", which is not
# what the refusal means: the entitlement is fine, the data is simply not out
# of its delay yet. Nothing here needs recent data, so nothing here asks.
FEED_DELAY = timedelta(minutes=15)


def last_rebuildable_session(now: datetime, close_time: time, zone: ZoneInfo) -> date:
    """The most recent date a session can honestly be rebuilt from.

    Today is excluded until it has closed and cleared the feed delay, for two
    separate reasons and either one would be enough. A session is labelled by
    where it settled, and a session still trading has not settled. And the
    delayed feed will not serve the current session's tape at all, so asking
    for it fails the whole walk rather than returning a thin day.
    """
    today = now.astimezone(zone).date()
    close_at = datetime.combine(today, close_time, tzinfo=zone)
    if now < close_at + FEED_DELAY:
        return today - timedelta(days=1)
    return today


def rebuild_window(
    gateway,
    config: Config,
    days: int,
    relative_spread: float,
    on_session=None,
) -> RebuiltWindow:
    """Rebuild every session in the trailing window, in calendar order.

    Shared by the training backfill and the replay so that both see exactly the
    same sessions built exactly the same way. A session whose ladder barely
    printed is dropped and counted rather than carried in thin.
    """
    zone = ZoneInfo(config.str_("session.timezone"))
    symbol = config.str_("underlying.symbol")
    entry_time = time.fromisoformat(config.str_("reconstruction.entry_time"))
    close_time = time.fromisoformat(config.str_("session.close_time"))
    minimum = config.float_("reconstruction.min_coverage")

    now, _ = gateway.clock()
    end = last_rebuildable_session(now, close_time, zone)
    start = end - timedelta(days=days)
    sessions = gateway.sessions(start, end)
    if not sessions:
        raise DataError(f"Alpaca's calendar lists no session between {start} and {end}")

    bars = gateway.minute_bars(
        symbol,
        datetime.combine(start, time(0, 0), tzinfo=zone),
        datetime.combine(end, time(23, 59), tzinfo=zone),
    )

    snapshots: list[ChainSnapshot] = []
    settlements: dict[date, float] = {}
    rebuilt: dict[date, ReconstructedSession] = {}
    coverages: list[float] = []
    thin = 0
    for session in sessions:
        day = session.session_date
        entry_at = datetime.combine(day, entry_time, tzinfo=zone)
        close_at = datetime.combine(day, close_time, tzinfo=zone)
        before_entry = bars[bars.index <= entry_at]
        before_close = bars[bars.index <= close_at]
        if before_entry.empty or before_close.empty:
            continue
        spot_at_entry = float(before_entry.iloc[-1]["close"])
        spot_at_close = float(before_close.iloc[-1]["close"])

        session_data = build(
            gateway, config, day, entry_at, close_at, spot_at_entry, spot_at_close,
            open_at=session.open_at,
        )
        coverages.append(session_data.coverage)
        if on_session is not None:
            on_session(session_data)
        if session_data.coverage < minimum:
            thin += 1
            continue
        snapshots.append(as_snapshot(session_data, entry_at, relative_spread))
        settlements[day] = spot_at_close
        rebuilt[day] = session_data

    return RebuiltWindow(
        snapshots=snapshots,
        settlements=settlements,
        sessions=rebuilt,
        coverages=coverages,
        dropped_thin=thin,
        modelled_relative_spread=relative_spread,
    )
