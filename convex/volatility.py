"""Black-Scholes, used only to read a volatility out of a price.

Nothing here prices a trade. Every quoted number this project acts on comes
from the book, and the one thing the book does not publish is the volatility
implied by it: Alpaca serves Greeks and implied volatility on every expiry
except the one this project trades, so on expiration day the volatility has to
be solved rather than read.

Both callers use this same code, which is the point of it living here. A
session rebuilt from the tape solves its volatilities through
convex.reconstruct, and a live 0DTE chain solves its own through
convex.features, and the two therefore mean the same quantity. Before this was
shared, the live path read the vendor's volatility while every training row
carried a solved one, so a model was fitted on one number and served another.
"""

from __future__ import annotations

import math

from convex.instruments import Right

_VOL_LOW = 1e-4
_VOL_HIGH = 5.0
_VOL_TOLERANCE = 1e-6
_MAX_ITERATIONS = 200


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes(
    spot: float, strike: float, right: Right, years: float, rate: float, vol: float
) -> float:
    """European price. Used only to invert a print into an implied volatility.

    SPY options are American, but early exercise on the last few hours of a
    contract's life is worth essentially nothing above intrinsic, and this is
    never used to price a trade, only to read a volatility out of a print.
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
