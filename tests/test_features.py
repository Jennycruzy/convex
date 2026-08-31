"""Feature arithmetic, and the refusals that keep look-ahead out."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from convex.errors import DataError
from convex.features import (
    build,
    gamma_exposure,
    integrated_variance,
    liquidity_features,
    realised_moments,
    lagged_results,
    time_to_close_years,
)
from convex.instruments import Right
from tests.conftest import build_test_chain

NOW = datetime.now(timezone.utc)
CLOSE = NOW + timedelta(hours=6)
RATE = 0.043  # config/convex.yaml, reconstruction.risk_free_rate


@pytest.fixture
def chain():
    return build_test_chain()


def test_time_to_close_refuses_a_past_close():
    with pytest.raises(DataError, match="not before"):
        time_to_close_years(CLOSE, NOW)


def test_integrated_variance_is_positive_and_scales_with_price(chain):
    tau = time_to_close_years(NOW, CLOSE)
    calls = [r for r in chain if r.contract.right is Right.CALL and r.contract.strike >= 650]
    variance = integrated_variance(calls, 650.0, tau)
    assert variance > 0.0
    # Doubling every premium doubles the integral, which is the linearity the
    # VIX construction relies on.
    from tests.conftest import build_test_chain as rebuild

    richer = rebuild(sigma=0.32)
    richer_calls = [r for r in richer if r.contract.right is Right.CALL and r.contract.strike >= 650]
    assert integrated_variance(richer_calls, 650.0, tau) > variance


def test_implied_skew_is_the_difference_between_the_two_sides(chain):
    features = build(chain, 650.0, NOW, CLOSE, np.linspace(-0.01, 0.01, 10), {}, rate=RATE)
    values = features.values
    assert values["implied_skew"] == pytest.approx(values["iv_up"] - values["iv_dn"])
    assert values["iv_total"] == pytest.approx(values["iv_up"] + values["iv_dn"])


def test_gamma_exposure_balance_is_bounded(chain):
    signed, absolute, balance = gamma_exposure(chain, 650.0)
    assert absolute > 0.0
    assert -1.0 <= balance <= 1.0
    assert abs(signed) <= absolute


def test_missing_greeks_raise_rather_than_defaulting(chain):
    from dataclasses import replace

    stripped = [replace(chain[0], greeks=None)] + list(chain[1:])
    with pytest.raises(DataError, match="no Greeks"):
        gamma_exposure(stripped, 650.0)


def test_liquidity_features_summarise_the_snapshot(chain):
    values = liquidity_features(chain)
    assert values["liq_half_spread"] > 0.0
    assert values["liq_relative_spread_p90"] >= values["liq_relative_spread"] * 0.5
    assert values["liq_tightness"] > 0.0


def test_realised_moments_need_history():
    with pytest.raises(DataError, match="at least two"):
        realised_moments([0.001])
    moments = realised_moments([0.01, -0.02, 0.005, -0.001, 0.004])
    assert moments["ret_lag1"] == pytest.approx(0.004)
    assert moments["rv_lag1"] == pytest.approx(0.004**2)


def test_a_family_with_no_history_reports_a_count_of_zero():
    assert lagged_results([]) == {
        "pnl_lag1": 0.0,
        "pnl_mean5": 0.0,
        "pnl_std5": 0.0,
        "pnl_count": 0.0,
    }


def test_feature_vector_names_are_stable_and_complete(chain):
    features = build(
        chain,
        650.0,
        NOW,
        CLOSE,
        np.linspace(-0.01, 0.01, 10),
        {"put_bwb": [12.0, -3.0], "straddle": []},
        rate=RATE,
    )
    for name in ("implied_skew", "gex_balance", "liq_tightness", "put_bwb_pnl_lag1"):
        assert name in features.values
    assert features.vector(["implied_skew", "iv_up"]).shape == (2,)
    with pytest.raises(DataError, match="missing"):
        features.vector(["not_a_feature"])


def _expiration_day_chain():
    """What Alpaca actually hands back on the day a contract expires.

    Measured 2026-08-31 on the live paper account: 0 of 124 contracts on that
    day's expiry carried Greeks or implied volatility, while every later expiry
    carried them on the same feed in the same call. The snapshot payload has no
    greeks key at all, so the rows arrive like this.
    """
    from dataclasses import replace

    # Priced at the same time to expiry the feature engine will compute from
    # NOW and CLOSE, so the volatilities solve back to what built them.
    return [
        replace(row, greeks=None, open_interest=None)
        for row in build_test_chain(tau=6.0 / (365.0 * 24.0))
    ]


def test_a_chain_with_no_greeks_still_builds_a_row():
    """The 31 August blocker, in a test.

    Before this, build raised on the first row and the agent never ran on the
    only expiry it trades.
    """
    features = build(
        _expiration_day_chain(), 650.0, NOW, CLOSE, np.linspace(-0.01, 0.01, 10), {}, rate=RATE
    )
    assert features.values["slope_up"] != 0.0
    assert features.values["slope_dn"] != 0.0
    assert features.values["iv_total"] > 0.0


def test_the_exposure_features_are_absent_rather_than_zero_without_greeks():
    """Absent, so anything that needs one says so instead of reading a zero."""
    features = build(
        _expiration_day_chain(), 650.0, NOW, CLOSE, np.linspace(-0.01, 0.01, 10), {}, rate=RATE
    )
    for name in ("gex_signed", "gex_absolute", "gex_balance"):
        assert name not in features.values
    with pytest.raises(DataError, match="missing"):
        features.vector(["gex_balance"])


def test_the_exposure_features_come_back_when_the_snapshot_carries_them(chain):
    """Absent only on the days they cannot be seen, present on every other."""
    features = build(chain, 650.0, NOW, CLOSE, np.linspace(-0.01, 0.01, 10), {}, rate=RATE)
    assert "gex_balance" in features.values


def test_the_slope_is_solved_not_read_so_a_wrong_vendor_greek_cannot_reach_it():
    """Stripping the Greeks off a chain does not move the slope by anything.

    This is what makes the live row and a rebuilt training row the same
    quantity: both solve the volatility out of the price, and neither reads the
    snapshot's own number.
    """
    from dataclasses import replace

    from convex.instruments import Greeks

    truthful = build_test_chain()
    lying = [
        replace(row, greeks=Greeks(0.5, 0.01, -50.0, 0.1, 0.0, 99.0)) for row in truthful
    ]
    honest = build(truthful, 650.0, NOW, CLOSE, np.linspace(-0.01, 0.01, 10), {}, rate=RATE)
    misled = build(lying, 650.0, NOW, CLOSE, np.linspace(-0.01, 0.01, 10), {}, rate=RATE)
    assert honest.values["slope_up"] == pytest.approx(misled.values["slope_up"])
    assert honest.values["slope_dn"] == pytest.approx(misled.values["slope_dn"])


def test_the_solving_rate_does_not_move_the_slope_at_zero_days():
    """The measurement that keeps the rate out of the live decision's read set.

    reconstruction.risk_free_rate is still a hypothesis, never checked against a
    curve. Solving the smile against it would make a live decision read an
    unmeasured number, which CalibrationGate exists to refuse. It does not,
    because at a few hours to expiry the discount factor is 0.99998 and the
    slope cannot tell the difference: across the whole plausible range the two
    slopes move by under one percent of their value. If that ever stops being
    true, this fails and the key belongs in CalibrationGate.REQUIRED.
    """
    chain = _expiration_day_chain()
    rows = [
        build(chain, 650.0, NOW, CLOSE, np.linspace(-0.01, 0.01, 10), {}, rate=rate)
        for rate in (0.0, 0.043, 0.10)
    ]
    for name in ("slope_up", "slope_dn"):
        values = [row.values[name] for row in rows]
        spread = max(values) - min(values)
        assert spread / abs(values[1]) < 0.01
