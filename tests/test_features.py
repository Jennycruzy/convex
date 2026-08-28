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
    features = build(chain, 650.0, NOW, CLOSE, np.linspace(-0.01, 0.01, 10), {})
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
    )
    for name in ("implied_skew", "gex_balance", "liq_tightness", "put_bwb_pnl_lag1"):
        assert name in features.values
    assert features.vector(["implied_skew", "iv_up"]).shape == (2,)
    with pytest.raises(DataError, match="missing"):
        features.vector(["not_a_feature"])
