from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from convex.config import load
from convex.errors import DataError
from convex.gap_continuation import signal
from scripts.profile_backtest import (
    Observation,
    Threshold,
    TradeOutcome,
    confidence_lower_bound,
    configured_baseline,
    direction_for_threshold,
    measure,
    select_threshold,
    signal_floor,
    threshold_grid,
    walk_forward,
)

ZONE = ZoneInfo("America/New_York")
BASELINE = Threshold(0.003, 0.0)


def trade(net: float, direction: int = 1) -> TradeOutcome:
    return TradeOutcome(
        direction=direction,
        description="test vertical",
        gross=net + 1.0,
        cost=1.0,
        net=net,
        net_edge=10.0,
        net_edge_lower_bound=5.0,
        rank=1,
    )


def observation(
    offset: int,
    *,
    gap: float | None = 0.01,
    vwap_distance: float | None = 0.005,
    direction: int = 1,
    net: float = 10.0,
) -> Observation:
    return Observation(
        session_date=date(2026, 1, 1) + timedelta(days=offset),
        gap=gap,
        signed_vwap_distance=vwap_distance,
        direction=direction if gap is not None else 0,
        trade=trade(net, direction) if gap is not None else None,
    )


def test_signal_can_require_a_minimum_distance_from_vwap():
    index = pd.date_range(datetime(2026, 9, 3, 9, 30, tzinfo=ZONE), periods=31, freq="min")
    frame = pd.DataFrame(
        {"open": [103.0] * 31, "close": [103.0] * 30 + [104.0], "volume": [10] * 31},
        index=index,
    )

    assert (
        signal(
            frame,
            prior_close=100.0,
            minimum_vwap_distance=0.005,
        )
        is not None
    )
    assert (
        signal(
            frame,
            prior_close=100.0,
            minimum_vwap_distance=0.01,
        )
        is None
    )


def test_negative_vwap_threshold_is_refused():
    index = pd.date_range(datetime(2026, 9, 3, 9, 30, tzinfo=ZONE), periods=31, freq="min")
    frame = pd.DataFrame(
        {"open": [103.0] * 31, "close": [103.0] * 30 + [104.0], "volume": [10] * 31},
        index=index,
    )

    with pytest.raises(DataError, match="VWAP distance"):
        signal(frame, prior_close=100.0, minimum_vwap_distance=-0.001)


def test_threshold_grid_always_contains_the_deployed_baseline():
    grid = threshold_grid((0.005,), (0.001,), BASELINE)
    assert BASELINE in grid
    assert Threshold(0.005, 0.001) in grid


def test_configured_baseline_matches_the_active_profile():
    assert configured_baseline(load()) == BASELINE


def test_signal_floor_covers_custom_thresholds_below_the_baseline():
    grid = threshold_grid((0.001,), (0.0,), BASELINE)
    assert signal_floor(grid) == Threshold(0.001, 0.0)


def test_threshold_measure_counts_no_signal_sessions_as_zero_pnl():
    rows = [
        observation(0, gap=None, vwap_distance=None),
        observation(1, net=12.0),
    ]

    result = measure(rows, BASELINE, confidence=0.95)

    assert result.sessions == 2
    assert result.signals == 1
    assert result.trades == 1
    assert result.net_total == pytest.approx(12.0)
    assert result.mean_net_per_session == pytest.approx(6.0)
    assert result.max_drawdown == pytest.approx(0.0)


def test_lower_bound_matches_the_one_sided_normal_convention():
    values = [10.0, 12.0, 8.0, 10.0]
    expected = sum(values) / len(values) - 1.6448536269514722 * (
        pd.Series(values).std(ddof=1) / len(values) ** 0.5
    )
    assert confidence_lower_bound(values, 0.95) == pytest.approx(expected)


def test_threshold_selection_refuses_a_positive_total_without_a_positive_bound():
    rows = [observation(index, net=-5.0) for index in range(20)]
    chosen = select_threshold(
        rows,
        thresholds=(BASELINE,),
        baseline=BASELINE,
        confidence=0.95,
        minimum_trades=20,
        minimum_lower_bound=0.01,
    )

    assert chosen.threshold == BASELINE
    assert not chosen.qualified
    assert "no threshold cleared" in chosen.reason


def test_direction_uses_signed_vwap_distance_and_preserves_side():
    up = observation(0, gap=0.01, vwap_distance=0.004, direction=1)
    down = observation(1, gap=-0.01, vwap_distance=-0.004, direction=-1)

    assert direction_for_threshold(up, Threshold(0.003, 0.003)) == 1
    assert direction_for_threshold(up, Threshold(0.003, 0.005)) == 0
    assert direction_for_threshold(down, Threshold(0.003, 0.003)) == -1


def test_walk_forward_does_not_use_the_current_session_for_selection():
    alternative = Threshold(0.01, 0.0)
    rows = [
        observation(0, gap=0.005, net=10.0),
        observation(1, gap=0.005, net=10.0),
        observation(2, gap=0.02, net=100.0),
        observation(3, gap=0.02, net=100.0),
        observation(4, gap=0.02, net=100.0),
    ]

    result = walk_forward(
        rows,
        thresholds=(BASELINE, alternative),
        baseline=BASELINE,
        confidence=0.95,
        minimum_training_sessions=2,
        minimum_training_trades=1,
        minimum_lower_bound=0.01,
    )

    # The alternative has two prior trades only when the fifth session is
    # selected. A selector that leaked the fourth session would switch one day
    # earlier.
    assert result.selected_thresholds[alternative.label()] == 1
    assert result.metrics.trades == 5
