"""The 10:00 ET feature snapshot.

Every feature here is computable from information available at the entry time
and nothing else. There is no look-ahead: the chain snapshot is taken at the
entry time, the realised moments are lagged by a full session, and the lagged
strategy results come from ledger records that were already written.

The features follow the research's specification:

  implied variance    integrated variance from the 0DTE chain to the close,
                      computed the way the VIX formula does, from out-of-the-
                      money options either side of the forward
  implied skew        the same integral run separately over the upside and the
                      downside, differenced. This is the highest-value feature
                      in the study, because realised skewness rather than
                      realised variance is what drives 0DTE payoffs
  slopes              how fast the smile rises away from the money, up and down
  lagged moments      yesterday's realised variance, skewness and return
  lagged results      each family's own last result, its five-day mean and its
                      five-day dispersion
  exposure proxies    open-interest and gamma weighted notional either side,
                      and the normalised balance between them
  liquidity           half-spread, displayed depth, relative spread, tightness

One honesty note that belongs in the write-up as much as in this docstring: the
exposure features are flow and exposure proxies built from traded volume, open
interest and leg Greeks. They are not a dealer-inventory reconstruction, and
nothing here should be read as knowing which side a dealer is on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np

from convex.errors import DataError
from convex.instruments import ChainEntry, Right

SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


@dataclass(frozen=True)
class FeatureSet:
    """One row of predictors, with the snapshot it was computed from."""

    taken_at: datetime
    spot: float
    time_to_close_years: float
    values: dict[str, float]

    def __post_init__(self) -> None:
        bad = {name: value for name, value in self.values.items() if not math.isfinite(value)}
        if bad:
            raise DataError(f"features are not finite: {bad}")

    def vector(self, names: Sequence[str]) -> np.ndarray:
        missing = [name for name in names if name not in self.values]
        if missing:
            raise DataError(f"feature snapshot is missing {missing}")
        return np.array([self.values[name] for name in names], dtype=float)

    def as_dict(self) -> dict[str, float]:
        return {name: round(value, 8) for name, value in self.values.items()}


def time_to_close_years(now: datetime, close: datetime) -> float:
    seconds = (close - now).total_seconds()
    if seconds <= 0.0:
        raise DataError(f"entry time {now} is not before the close {close}")
    return seconds / SECONDS_PER_YEAR


def _otm(chain: Iterable[ChainEntry], spot: float, right: Right) -> list[ChainEntry]:
    if right is Right.CALL:
        rows = [row for row in chain if row.contract.right is right and row.contract.strike >= spot]
    else:
        rows = [row for row in chain if row.contract.right is right and row.contract.strike <= spot]
    return sorted(rows, key=lambda row: row.contract.strike)


def _strike_widths(strikes: Sequence[float]) -> list[float]:
    """Half the distance to each neighbour, the VIX formula's delta-K."""
    if len(strikes) < 2:
        raise DataError("integrated variance needs at least two strikes on a side")
    widths = []
    for index, strike in enumerate(strikes):
        if index == 0:
            widths.append(strikes[1] - strike)
        elif index == len(strikes) - 1:
            widths.append(strike - strikes[-2])
        else:
            widths.append((strikes[index + 1] - strikes[index - 1]) / 2.0)
    return widths


def integrated_variance(rows: Sequence[ChainEntry], spot: float, tau: float) -> float:
    """Contribution to VIX-style integrated variance from one side of the chain.

    The discount factor is dropped rather than assumed: over a six-hour horizon
    at any plausible rate it moves the result by less than a basis point of
    variance, and inventing a rate would be a number the agent did not measure.
    """
    if tau <= 0.0:
        raise DataError(f"time to expiry must be positive, found {tau}")
    strikes = [row.contract.strike for row in rows]
    widths = _strike_widths(strikes)
    total = 0.0
    for row, width in zip(rows, widths):
        strike = row.contract.strike
        if strike <= 0.0:
            raise DataError(f"{row.contract.symbol}: non-positive strike {strike}")
        total += (width / strike**2) * row.quote.mid
    return (2.0 / tau) * total


def gamma_exposure(rows: Iterable[ChainEntry], spot: float) -> tuple[float, float, float]:
    """Signed, absolute and balanced open-interest gamma notional.

    Calls are counted positive and puts negative, which is a convention rather
    than a claim about dealer positioning. The balance is normalised so it stays
    comparable across days with very different open interest.
    """
    signed = 0.0
    absolute = 0.0
    for row in rows:
        greeks = row.require_greeks()
        open_interest = row.require_open_interest()
        exposure = open_interest * greeks.gamma * 100.0 * spot**2
        sign = 1.0 if row.contract.right is Right.CALL else -1.0
        signed += sign * exposure
        absolute += abs(exposure)
    balance = signed / (absolute + 1.0)
    return signed, absolute, balance


def liquidity_features(rows: Sequence[ChainEntry]) -> dict[str, float]:
    """Half-spread, depth, relative spread and tightness across the snapshot."""
    if not rows:
        raise DataError("liquidity features need a non-empty chain snapshot")
    half_spreads = np.array([row.quote.half_spread for row in rows])
    depths = np.array([min(row.quote.bid_size, row.quote.ask_size) for row in rows], dtype=float)
    relative = np.array([row.quote.relative_spread for row in rows])
    tightness = half_spreads / (depths + 1.0)
    return {
        "liq_half_spread": float(half_spreads.mean()),
        "liq_depth": float(depths.mean()),
        "liq_relative_spread": float(relative.mean()),
        "liq_relative_spread_p90": float(np.quantile(relative, 0.9)),
        "liq_tightness": float(tightness.mean()),
    }


def traded_volume(value: int | None) -> float:
    """A contract's session volume, reading absence as zero rather than defaulting.

    Law 3 forbids a missing field becoming a zero, and this is the one place the
    distinction genuinely collapses: Alpaca omits the daily bar for a contract
    that has not traded today, so the absence *is* the observation that nothing
    printed. That is a reading of the feed's shape, not a fallback, and it is
    written here once so nothing downstream has to decide it again.
    """
    return 0.0 if value is None else float(value)


def tape_features(rows: Sequence[tuple[Right, int | None]]) -> dict[str, float]:
    """Liquidity and flow read off the tape instead of off the book.

    The book is what the cost model wants and it is exactly what a rebuilt
    session cannot have, since Alpaca keeps no historical option quotes. What it
    does keep is what traded. These four are computable identically from a live
    snapshot and from the tape of a session that ended a year ago, which is the
    property that makes them usable: a model fitted on history can be run at
    10:00 without the features shifting meaning underneath it.

      tape_volume         how much traded across the chain, logged because the
                          raw figure spans orders of magnitude between a quiet
                          Tuesday and an expiry with news in it
      tape_breadth        the share of the chain that traded at all, which is
                          the closest honest proxy for how far a structure can
                          reach for strikes before it runs out of liquidity
      tape_concentration  a Herfindahl over volume share. One busy strike and a
                          dead ladder is a different market from an evenly
                          traded one, and the two have very different execution
                          costs at the wings
      tape_put_share      puts as a share of traded volume. This is the flow
                          counterpart to implied skew, and the research is that
                          skew drives 0DTE results, so a directional imbalance
                          on the tape belongs in the row beside it
    """
    if not rows:
        raise DataError("tape features need a non-empty chain snapshot")
    volumes = [traded_volume(volume) for _, volume in rows]
    total = sum(volumes)
    traded = sum(1 for volume in volumes if volume > 0.0)
    put_volume = sum(
        volume for (right, _), volume in zip(rows, volumes) if right is Right.PUT
    )
    shares = [volume / total for volume in volumes] if total > 0.0 else []
    return {
        "tape_volume": float(np.log1p(total)),
        "tape_breadth": traded / len(rows),
        "tape_concentration": float(sum(share * share for share in shares)),
        # With nothing traded there is no imbalance to report, and a half is the
        # only value that asserts none. It is reached from an observed total of
        # zero rather than from a missing field.
        "tape_put_share": (put_volume / total) if total > 0.0 else 0.5,
    }


def realised_moments(log_returns: Sequence[float]) -> dict[str, float]:
    """Lagged realised variance, skewness and return from prior sessions."""
    series = np.asarray(list(log_returns), dtype=float)
    if series.size < 2:
        raise DataError("realised moments need at least two prior sessions")
    sigma = float(series.std(ddof=1))
    if sigma <= 0.0:
        raise DataError("prior sessions show zero dispersion")
    centred = series - series.mean()
    return {
        "rv_lag1": float(series[-1] ** 2),
        "ret_lag1": float(series[-1]),
        "rv_5": float(np.mean(series[-5:] ** 2)),
        "rskew": float(np.mean(centred**3) / sigma**3),
    }


def lagged_results(pnl_history: Sequence[float]) -> dict[str, float]:
    """A family's own last result, five-day mean and five-day dispersion.

    A family with no history yet contributes zeros, and that is not a silent
    default: it is the honest encoding of "no prior result", and the classifier
    is told about it explicitly through the accompanying count feature.
    """
    series = np.asarray(list(pnl_history), dtype=float)
    if series.size == 0:
        return {"pnl_lag1": 0.0, "pnl_mean5": 0.0, "pnl_std5": 0.0, "pnl_count": 0.0}
    window = series[-5:]
    return {
        "pnl_lag1": float(series[-1]),
        "pnl_mean5": float(window.mean()),
        "pnl_std5": float(window.std(ddof=1)) if window.size > 1 else 0.0,
        "pnl_count": float(series.size),
    }


def build(
    chain: Sequence[ChainEntry],
    spot: float,
    now: datetime,
    close: datetime,
    prior_returns: Sequence[float],
    family_pnl: dict[str, Sequence[float]],
) -> FeatureSet:
    """Assemble the full predictor row from one chain snapshot."""
    tau = time_to_close_years(now, close)
    calls = _otm(chain, spot, Right.CALL)
    puts = _otm(chain, spot, Right.PUT)
    if len(calls) < 2 or len(puts) < 2:
        raise DataError(
            f"chain snapshot has {len(calls)} out-of-the-money calls and {len(puts)} puts; "
            "integrated variance needs at least two on each side"
        )

    variance_up = integrated_variance(calls, spot, tau)
    variance_dn = integrated_variance(puts, spot, tau)
    signed_gamma, absolute_gamma, gamma_balance = gamma_exposure(chain, spot)

    values: dict[str, float] = {
        "iv_total": variance_up + variance_dn,
        "iv_up": variance_up,
        "iv_dn": variance_dn,
        "implied_skew": variance_up - variance_dn,
        "slope_up": _smile_slope(calls, spot),
        "slope_dn": _smile_slope(puts, spot),
        "gex_signed": signed_gamma,
        "gex_absolute": absolute_gamma,
        "gex_balance": gamma_balance,
    }
    values.update(liquidity_features(chain))
    values.update(
        tape_features([(row.contract.right, row.volume) for row in chain])
    )
    values.update(realised_moments(prior_returns))
    for family, history in sorted(family_pnl.items()):
        for name, value in lagged_results(history).items():
            values[f"{family}_{name}"] = value

    return FeatureSet(taken_at=now, spot=spot, time_to_close_years=tau, values=values)


def _smile_slope(rows: Sequence[ChainEntry], spot: float) -> float:
    """Least-squares slope of implied volatility against log-moneyness."""
    usable = [row for row in rows if row.greeks is not None]
    if len(usable) < 3:
        raise DataError(
            f"smile slope needs at least three contracts with Greeks, found {len(usable)}"
        )
    moneyness = np.array([math.log(row.contract.strike / spot) for row in usable])
    vols = np.array([row.require_greeks().implied_volatility for row in usable])
    slope, _ = np.polyfit(moneyness, vols, 1)
    return float(slope)
