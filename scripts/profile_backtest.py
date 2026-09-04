"""Walk-forward research for the active gap-continuation vertical profile.

This is deliberately separate from the scheduled runner. It rebuilds closed
sessions from Alpaca's historical option prints, uses only stock returns that
were available before each session to rank the vertical, and tests signal
thresholds without writing the live configuration, ledger, or placing orders.

Historical option books are unavailable. The spread supplied to this command
is therefore modelled uniformly from each option print and must be reported
with every result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from convex import backtest, reconstruct, scenarios
from convex.agent import PricedCandidate, rank
from convex.config import Config, load
from convex.costs import CostModel
from convex.data.alpaca import AlpacaGateway, MarketSession
from convex.edge import evaluate
from convex.errors import ConvexError, DataError, UndefinedRiskError
from convex.gap_continuation import signal
from convex.sizing import PortfolioState, size_position
from convex.structures import build_candidates
from convex.structures.base import Family
from convex.training import settlement_pnl_of

PROFILE = "gap_continuation_vertical"
DEFAULT_GAPS = (0.003, 0.004, 0.005, 0.0075)
DEFAULT_VWAP_DISTANCES = (0.0, 0.0005, 0.001, 0.002)


@dataclass(frozen=True, order=True)
class Threshold:
    """A signal threshold, expressed as fractions rather than SPY dollars."""

    minimum_gap: float
    minimum_vwap_distance: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_gap) or self.minimum_gap <= 0.0:
            raise DataError(f"minimum gap must be finite and positive, found {self.minimum_gap}")
        if not math.isfinite(self.minimum_vwap_distance) or self.minimum_vwap_distance < 0.0:
            raise DataError(
                "minimum VWAP distance must be finite and non-negative, "
                f"found {self.minimum_vwap_distance}"
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "minimum_gap": self.minimum_gap,
            "minimum_vwap_distance": self.minimum_vwap_distance,
        }

    def label(self) -> str:
        return f"gap={self.minimum_gap:.4f}, vwap={self.minimum_vwap_distance:.4f}"


@dataclass(frozen=True)
class TradeOutcome:
    """The top candidate's reconstructed expiry result after modelled costs."""

    direction: int
    description: str
    gross: float
    cost: float
    net: float
    net_edge: float
    net_edge_lower_bound: float
    rank: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "direction": self.direction,
            "description": self.description,
            "gross": round(self.gross, 2),
            "cost": round(self.cost, 2),
            "net": round(self.net, 2),
            "net_edge": round(self.net_edge, 2),
            "net_edge_lower_bound": round(self.net_edge_lower_bound, 2),
            "rank": self.rank,
        }


@dataclass(frozen=True)
class Observation:
    """One usable reconstructed session, including a possible no-trade day."""

    session_date: date
    gap: float | None
    signed_vwap_distance: float | None
    direction: int
    trade: TradeOutcome | None


@dataclass(frozen=True)
class Metrics:
    """P&L statistics for one fixed threshold or a walk-forward path."""

    threshold: Threshold | None
    sessions: int
    signals: int
    trades: int
    gross_total: float
    net_total: float
    cost_total: float
    mean_net_per_session: float | None
    mean_net_per_trade: float | None
    hit_rate: float | None
    lower_bound: float | None
    gross_sharpe: float | None
    net_sharpe: float | None
    expected_shortfall: float | None
    max_drawdown: float

    def as_dict(self) -> dict:
        return {
            "threshold": None if self.threshold is None else self.threshold.as_dict(),
            "sessions": self.sessions,
            "signals": self.signals,
            "trades": self.trades,
            "gross_total": round(self.gross_total, 2),
            "net_total": round(self.net_total, 2),
            "cost_total": round(self.cost_total, 2),
            "mean_net_per_session": (
                None if self.mean_net_per_session is None else round(self.mean_net_per_session, 2)
            ),
            "mean_net_per_trade": (
                None if self.mean_net_per_trade is None else round(self.mean_net_per_trade, 2)
            ),
            "hit_rate": None if self.hit_rate is None else round(self.hit_rate, 4),
            "lower_bound": None if self.lower_bound is None else round(self.lower_bound, 2),
            "gross_sharpe": (None if self.gross_sharpe is None else round(self.gross_sharpe, 3)),
            "net_sharpe": None if self.net_sharpe is None else round(self.net_sharpe, 3),
            "expected_shortfall": (
                None if self.expected_shortfall is None else round(self.expected_shortfall, 2)
            ),
            "max_drawdown": round(self.max_drawdown, 2),
        }


@dataclass(frozen=True)
class Selection:
    """A threshold selected without looking at the day it will be applied to."""

    threshold: Threshold
    qualified: bool
    reason: str
    training_metrics: Metrics

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold.as_dict(),
            "qualified": self.qualified,
            "reason": self.reason,
            "training_metrics": self.training_metrics.as_dict(),
        }


@dataclass(frozen=True)
class WalkForwardResult:
    """The result of selecting thresholds on strictly earlier sessions."""

    metrics: Metrics
    warmup_sessions: int
    fallback_sessions: int
    selected_thresholds: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "metrics": self.metrics.as_dict(),
            "warmup_sessions": self.warmup_sessions,
            "fallback_sessions": self.fallback_sessions,
            "selected_thresholds": self.selected_thresholds,
        }


def confidence_lower_bound(values: Sequence[float], confidence: float) -> float | None:
    """One-sided normal lower bound using the same convention as edge.py."""
    if not 0.5 < confidence < 1.0:
        raise DataError(f"confidence must lie strictly between 0.5 and 1, found {confidence}")
    array = np.asarray(list(values), dtype=float)
    if not np.isfinite(array).all():
        raise DataError("trade outcomes must be finite")
    if array.size < 2:
        return None
    standard_error = float(array.std(ddof=1)) / sqrt(array.size)
    return float(array.mean() - NormalDist().inv_cdf(confidence) * standard_error)


def _summarize(
    threshold: Threshold | None,
    sessions: int,
    signals: int,
    gross: Sequence[float],
    costs: Sequence[float],
    net: Sequence[float],
    confidence: float,
    daily_gross: Sequence[float] | None = None,
    daily_net: Sequence[float] | None = None,
) -> Metrics:
    gross_values = list(gross)
    cost_values = list(costs)
    net_values = list(net)
    session_gross = list(daily_gross) if daily_gross is not None else gross_values
    session_net = list(daily_net) if daily_net is not None else net_values
    if len(gross_values) != len(cost_values) or len(gross_values) != len(net_values):
        raise DataError("gross, cost, and net outcomes must have equal lengths")
    if sessions < 0 or signals < 0 or signals > sessions:
        raise DataError(f"invalid session counts: sessions={sessions}, signals={signals}")
    if len(session_gross) != sessions or len(session_net) != sessions:
        raise DataError("daily outcome series must contain one value per session")
    net_array = np.asarray(session_net, dtype=float)

    return Metrics(
        threshold=threshold,
        sessions=sessions,
        signals=signals,
        trades=len(net_values),
        gross_total=float(sum(gross_values)),
        net_total=float(sum(net_values)),
        cost_total=float(sum(cost_values)),
        mean_net_per_session=(float(sum(net_values) / sessions) if sessions else None),
        mean_net_per_trade=(float(sum(net_values) / len(net_values)) if net_values else None),
        hit_rate=(
            float(sum(value > 0.0 for value in net_values) / len(net_values))
            if net_values
            else None
        ),
        lower_bound=confidence_lower_bound(net_values, confidence),
        gross_sharpe=backtest.sharpe(np.asarray(session_gross, dtype=float)),
        net_sharpe=backtest.sharpe(net_array),
        expected_shortfall=(
            backtest.expected_shortfall(net_array, 0.01) if net_array.size else None
        ),
        max_drawdown=backtest.max_drawdown(net_array),
    )


def direction_for_threshold(observation: Observation, threshold: Threshold) -> int:
    """Apply the same inequalities as gap_continuation.signal."""
    if observation.gap is None or observation.signed_vwap_distance is None:
        return 0
    if (
        observation.direction > 0
        and observation.gap >= threshold.minimum_gap
        and observation.signed_vwap_distance > threshold.minimum_vwap_distance
    ):
        return 1
    if (
        observation.direction < 0
        and observation.gap <= -threshold.minimum_gap
        and observation.signed_vwap_distance < -threshold.minimum_vwap_distance
    ):
        return -1
    return 0


def measure(
    observations: Sequence[Observation], threshold: Threshold, confidence: float
) -> Metrics:
    """Measure a threshold with zero P&L on sessions it does not select."""
    gross: list[float] = []
    costs: list[float] = []
    net: list[float] = []
    daily_gross: list[float] = []
    daily_net: list[float] = []
    signals = 0

    for observation in observations:
        direction = direction_for_threshold(observation, threshold)
        if direction:
            signals += 1
        trade = observation.trade if direction == observation.direction else None
        if trade is None:
            daily_gross.append(0.0)
            daily_net.append(0.0)
            continue
        gross.append(trade.gross)
        costs.append(trade.cost)
        net.append(trade.net)
        daily_gross.append(trade.gross)
        daily_net.append(trade.net)

    return _summarize(
        threshold,
        len(observations),
        signals,
        gross,
        costs,
        net,
        confidence,
        daily_gross,
        daily_net,
    )


def select_threshold(
    observations: Sequence[Observation],
    thresholds: Sequence[Threshold],
    baseline: Threshold,
    confidence: float,
    minimum_trades: int,
    minimum_lower_bound: float,
) -> Selection:
    """Select only a threshold whose prior results clear the promotion bar."""
    results = [measure(observations, threshold, confidence) for threshold in thresholds]
    eligible = [
        result
        for result in results
        if result.trades >= minimum_trades and result.lower_bound is not None
    ]
    qualified = [result for result in eligible if result.lower_bound > minimum_lower_bound]
    if qualified:
        chosen = max(
            qualified,
            key=lambda result: (
                result.lower_bound,
                result.mean_net_per_session or float("-inf"),
                result.net_total,
                -result.threshold.minimum_gap,
                -result.threshold.minimum_vwap_distance,
            ),
        )
        return Selection(chosen.threshold, True, "training lower bound cleared", chosen)

    baseline_metrics = measure(observations, baseline, confidence)
    reason = (
        f"no threshold cleared the {confidence:.0%} lower-bound requirement "
        f"of {minimum_lower_bound:.2f} with at least {minimum_trades} trades"
    )
    return Selection(baseline, False, reason, baseline_metrics)


def walk_forward(
    observations: Sequence[Observation],
    thresholds: Sequence[Threshold],
    baseline: Threshold,
    confidence: float,
    minimum_training_sessions: int,
    minimum_training_trades: int,
    minimum_lower_bound: float,
) -> WalkForwardResult:
    """Choose a threshold from sessions strictly before each applied session."""
    gross: list[float] = []
    costs: list[float] = []
    net: list[float] = []
    daily_gross: list[float] = []
    daily_net: list[float] = []
    selected = Counter()
    warmup = 0
    fallback = 0
    signals = 0

    for index, observation in enumerate(observations):
        if index < minimum_training_sessions:
            threshold = baseline
            warmup += 1
        else:
            choice = select_threshold(
                observations[:index],
                thresholds,
                baseline,
                confidence,
                minimum_training_trades,
                minimum_lower_bound,
            )
            threshold = choice.threshold
            if not choice.qualified:
                fallback += 1
        selected[threshold.label()] += 1

        direction = direction_for_threshold(observation, threshold)
        if direction:
            signals += 1
        trade = observation.trade if direction == observation.direction else None
        if trade is None:
            daily_gross.append(0.0)
            daily_net.append(0.0)
            continue
        gross.append(trade.gross)
        costs.append(trade.cost)
        net.append(trade.net)
        daily_gross.append(trade.gross)
        daily_net.append(trade.net)

    return WalkForwardResult(
        metrics=_summarize(
            None,
            len(observations),
            signals,
            gross,
            costs,
            net,
            confidence,
            daily_gross,
            daily_net,
        ),
        warmup_sessions=warmup,
        fallback_sessions=fallback,
        selected_thresholds=dict(sorted(selected.items())),
    )


def _parse_values(raw: str, name: str) -> tuple[float, ...]:
    values: list[float] = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            parsed = float(value)
        except ValueError as error:
            raise DataError(f"{name} contains {value!r}, which is not a number") from error
        if not math.isfinite(parsed):
            raise DataError(f"{name} contains non-finite value {value!r}")
        values.append(parsed)
    if not values:
        raise DataError(f"{name} must contain at least one value")
    return tuple(values)


def _configured_time(config: Config, key: str) -> time:
    raw = config.str_(key)
    try:
        return time.fromisoformat(raw)
    except ValueError as error:
        raise DataError(f"{key} must be an ISO time, found {raw!r}") from error


def configured_baseline(config: Config) -> Threshold:
    return Threshold(
        config.float_("strategy.gap_continuation.minimum_gap"),
        config.float_("strategy.gap_continuation.minimum_vwap_distance"),
    )


def signal_floor(thresholds: Sequence[Threshold]) -> Threshold:
    if not thresholds:
        raise DataError("threshold grid must contain at least one threshold")
    return Threshold(
        min(threshold.minimum_gap for threshold in thresholds),
        min(threshold.minimum_vwap_distance for threshold in thresholds),
    )


def threshold_grid(
    gaps: Sequence[float], vwap_distances: Sequence[float], baseline: Threshold
) -> tuple[Threshold, ...]:
    """Add the configured baseline so every run has a direct comparison."""
    thresholds = {Threshold(gap, distance) for gap in gaps for distance in vwap_distances}
    thresholds.add(baseline)
    return tuple(sorted(thresholds))


def _regular_day(frame: pd.DataFrame, day: date, final_time: time) -> pd.DataFrame:
    return frame[
        (frame.index.date == day)
        & (frame.index.time >= time(9, 30))
        & (frame.index.time <= final_time)
    ]


def _asof_scenarios(
    full: scenarios.ScenarioSet,
    session_day: date,
    config: Config,
    zone: ZoneInfo,
    minimum_sessions: int,
) -> scenarios.ScenarioSet | None:
    oldest = session_day - timedelta(days=config.int_("data.scenario_lookback_days"))
    prior = [
        (day, float(value))
        for day, value in zip(full.source_days, full.log_returns)
        if oldest <= day < session_day
    ]
    if len(prior) < minimum_sessions:
        return None
    return scenarios.ScenarioSet(
        log_returns=np.asarray([value for _, value in prior], dtype=float),
        source_days=tuple(day for day, _ in prior),
        entry_time=full.entry_time,
        exit_time=full.exit_time,
        volatility_scale=1.0,
        built_at=datetime.combine(
            session_day, _configured_time(config, "session.entry_time"), tzinfo=zone
        ),
    )


def _trade_for_direction(
    snapshot: reconstruct.ChainSnapshot,
    direction: int,
    settlement: float,
    scenario_set: scenarios.ScenarioSet,
    profile_config: Config,
    config: Config,
    cost_model: CostModel,
    equity: float,
) -> TradeOutcome | None:
    prefix = "bull call" if direction > 0 else "bear put"
    built = build_candidates(
        snapshot.entries,
        profile_config,
        snapshot.spot,
        families=(Family.DEBIT_VERTICAL,),
    )
    candidates = [
        candidate
        for candidate in built.get(Family.DEBIT_VERTICAL, [])
        if candidate.description.startswith(prefix)
    ]
    priced: list[PricedCandidate] = []
    for candidate in candidates:
        try:
            estimate = evaluate(
                candidate.legs,
                scenario_set,
                cost_model,
                snapshot.spot,
                1,
                config.float_("risk.es_confidence"),
            )
        except UndefinedRiskError:
            continue
        priced.append(PricedCandidate(candidate, estimate))

    ordered = rank(priced, [str(Family.DEBIT_VERTICAL)])
    portfolio = PortfolioState(equity, equity, 0, 0.0)
    minimum_edge = config.float_("risk.min_net_edge_dollars")
    minimum_bound = config.float_("risk.min_net_edge_lower_bound_dollars")
    confidence = config.float_("risk.net_edge_confidence_level")

    for candidate_rank, priced_candidate in enumerate(
        ordered[: config.int_("candidates.max_ranked_attempts")], start=1
    ):
        estimate = priced_candidate.estimate
        if estimate.net_edge < minimum_edge:
            continue
        lower_bound = estimate.net_edge_lower_bound(confidence)
        if lower_bound <= minimum_bound:
            continue
        size = size_position(estimate, portfolio, config)
        if not size.trades:
            continue
        candidate = priced_candidate.candidate
        gross = settlement_pnl_of(
            candidate,
            cost_model.mid_debit(candidate.legs),
            settlement,
        )
        net = gross - estimate.cost.total
        return TradeOutcome(
            direction=direction,
            description=candidate.description,
            gross=gross,
            cost=estimate.cost.total,
            net=net,
            net_edge=estimate.net_edge,
            net_edge_lower_bound=lower_bound,
            rank=candidate_rank,
        )
    return None


def build_observations(
    window: reconstruct.RebuiltWindow,
    stock_bars: pd.DataFrame,
    calendar: Sequence[MarketSession],
    config: Config,
    signal_threshold: Threshold,
    minimum_scenario_sessions: int,
    equity: float,
) -> tuple[list[Observation], dict[str, int]]:
    """Build point-in-time opportunities from one reconstructed window."""
    if not window.snapshots:
        raise DataError("the rebuild produced no usable snapshots")

    zone = ZoneInfo(config.str_("session.timezone"))
    entry_time = _configured_time(config, "session.entry_time")
    close_time = _configured_time(config, "session.close_time")
    if entry_time > close_time:
        raise DataError("session.entry_time must not be after session.close_time")
    local = (
        stock_bars.tz_convert(zone)
        if stock_bars.index.tz is not None
        else stock_bars.tz_localize("UTC").tz_convert(zone)
    )
    full_scenarios = scenarios.build_from_bars(stock_bars, config)
    sessions = sorted(window.snapshots, key=lambda snapshot: snapshot.session_date)
    calendar_days = sorted(session.session_date for session in calendar)
    previous_by_day: dict[date, date] = {}
    for index, day in enumerate(calendar_days):
        if index:
            previous_by_day[day] = calendar_days[index - 1]

    cost_model = CostModel.from_config(config)
    profile_config = config.with_overrides(
        {
            "structures.enabled": [Family.DEBIT_VERTICAL.value],
            "structures.tie_break_order": [Family.DEBIT_VERTICAL.value],
        }
    )
    observations: list[Observation] = []
    skipped: Counter[str] = Counter()

    for snapshot in sessions:
        day = snapshot.session_date
        day_bars = local[local.index.date == day]
        prior_day = previous_by_day.get(day)
        prior_bars = (
            _regular_day(local, prior_day, close_time) if prior_day is not None else pd.DataFrame()
        )
        if day_bars.empty or prior_bars.empty:
            skipped["underlying_bars"] += 1
            continue
        prior_close = float(prior_bars["close"].iloc[-1])
        try:
            found = signal(
                day_bars,
                prior_close,
                minimum_gap=signal_threshold.minimum_gap,
                signal_time=entry_time,
                minimum_vwap_distance=signal_threshold.minimum_vwap_distance,
            )
        except DataError:
            skipped["signal_data"] += 1
            continue

        if found is None:
            observations.append(Observation(day, None, None, 0, None))
            continue

        scenario_set = _asof_scenarios(
            full_scenarios,
            day,
            config,
            zone,
            minimum_scenario_sessions,
        )
        if scenario_set is None:
            skipped["scenario_warmup"] += 1
            continue

        signed_vwap_distance = found.price_at_signal / found.vwap - 1.0
        try:
            trade = _trade_for_direction(
                snapshot,
                found.direction,
                window.settlements[day],
                scenario_set,
                profile_config,
                config,
                cost_model,
                equity,
            )
        except DataError:
            skipped["candidate_data"] += 1
            trade = None
        observations.append(
            Observation(
                session_date=day,
                gap=found.gap,
                signed_vwap_distance=signed_vwap_distance,
                direction=found.direction,
                trade=trade,
            )
        )

    return observations, dict(sorted(skipped.items()))


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.2f}"


def _bound(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.2f}"


def _print_metrics(prefix: str, metrics: Metrics, confidence: float) -> None:
    print(
        f"{prefix}: sessions={metrics.sessions} signals={metrics.signals} "
        f"trades={metrics.trades} gross={_money(metrics.gross_total)} "
        f"net={_money(metrics.net_total)} cost={_money(metrics.cost_total)} "
        f"{confidence:.0%} LB={_bound(metrics.lower_bound)} drawdown={_money(metrics.max_drawdown)}"
    )


def _selection_results(
    observations: Sequence[Observation],
    thresholds: Sequence[Threshold],
    holdout_sessions: int,
    confidence: float,
) -> list[dict]:
    train = observations[:-holdout_sessions]
    holdout = observations[-holdout_sessions:]
    results: list[dict] = []
    for threshold in thresholds:
        train_metrics = measure(train, threshold, confidence)
        holdout_metrics = measure(holdout, threshold, confidence)
        results.append(
            {
                "threshold": threshold.as_dict(),
                "train": train_metrics.as_dict(),
                "holdout": holdout_metrics.as_dict(),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument(
        "--relative-spread",
        type=float,
        default=None,
        help="modelled historical spread per leg; defaults to the hard admission cap",
    )
    parser.add_argument(
        "--gap-thresholds",
        default=",".join(str(value) for value in DEFAULT_GAPS),
        help="comma-separated minimum overnight gaps, as fractions",
    )
    parser.add_argument(
        "--vwap-thresholds",
        default=",".join(str(value) for value in DEFAULT_VWAP_DISTANCES),
        help="comma-separated minimum signed distance from VWAP, as fractions",
    )
    parser.add_argument("--holdout-sessions", type=int, default=60)
    parser.add_argument("--min-training-sessions", type=int, default=60)
    parser.add_argument("--min-training-trades", type=int, default=20)
    parser.add_argument("--min-holdout-trades", type=int, default=10)
    parser.add_argument("--min-scenario-sessions", type=int, default=60)
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--json", type=Path, help="write the research report here")
    arguments = parser.parse_args()

    if arguments.days <= 0:
        raise DataError(f"--days must be positive, found {arguments.days}")
    if arguments.holdout_sessions <= 0 or arguments.holdout_sessions >= arguments.days:
        raise DataError("--holdout-sessions must be positive and smaller than --days")
    if arguments.min_training_sessions < 0:
        raise DataError("--min-training-sessions must not be negative")
    if arguments.min_training_trades < 1 or arguments.min_holdout_trades < 1:
        raise DataError("minimum trade counts must be positive")
    if arguments.min_scenario_sessions < 2:
        raise DataError("--min-scenario-sessions must be at least two")
    if arguments.equity <= 0.0:
        raise DataError("--equity must be positive")

    config = load()
    if config.str_("strategy.active_profile") != PROFILE:
        raise DataError(
            f"active profile is {config.str_('strategy.active_profile')!r}, expected {PROFILE!r}"
        )

    gaps = _parse_values(arguments.gap_thresholds, "--gap-thresholds")
    vwap_distances = _parse_values(arguments.vwap_thresholds, "--vwap-thresholds")
    baseline = configured_baseline(config)
    thresholds = threshold_grid(gaps, vwap_distances, baseline)
    signal_threshold = signal_floor(thresholds)
    if arguments.relative_spread is None:
        relative_spread = config.float_("liquidity.admission_spread_cap")
    else:
        relative_spread = arguments.relative_spread
    if not math.isfinite(relative_spread) or relative_spread < 0.0:
        raise DataError(
            f"--relative-spread must be finite and non-negative, found {relative_spread}"
        )

    zone = ZoneInfo(config.str_("session.timezone"))
    print(
        f"research only: no orders, no ledger writes, no live config changes; "
        f"modelled spread {relative_spread:.4f} per leg"
    )
    print(f"baseline: {baseline.label()} (configured live profile)")
    print(f"signal floor used for reconstruction: {signal_threshold.label()}")

    with AlpacaGateway(config) as gateway:
        window = reconstruct.rebuild_window(gateway, config, arguments.days, relative_spread)
        if not window.snapshots:
            raise DataError("no historical option sessions were rebuildable")
        first = min(snapshot.session_date for snapshot in window.snapshots)
        last = max(snapshot.session_date for snapshot in window.snapshots)
        start = first - timedelta(days=config.int_("data.scenario_lookback_days") + 10)
        stock_bars = gateway.minute_bars(
            config.str_("underlying.symbol"),
            datetime.combine(start, time(0, 0), tzinfo=zone),
            datetime.combine(last, time(23, 59), tzinfo=zone),
        )
        calendar = gateway.sessions(start, last)
        observations, skipped = build_observations(
            window,
            stock_bars,
            calendar,
            config,
            signal_threshold,
            arguments.min_scenario_sessions,
            arguments.equity,
        )

    observations.sort(key=lambda observation: observation.session_date)
    if len(observations) <= arguments.holdout_sessions:
        raise DataError(
            f"only {len(observations)} usable observations; need more than "
            f"{arguments.holdout_sessions} for the holdout"
        )

    holdout = arguments.holdout_sessions
    train = observations[:-holdout]
    test = observations[-holdout:]
    confidence = config.float_("risk.net_edge_confidence_level")
    minimum_lower_bound = config.float_("risk.min_net_edge_lower_bound_dollars")
    rows = _selection_results(observations, thresholds, holdout, confidence)
    selection = select_threshold(
        train,
        thresholds,
        baseline,
        confidence,
        arguments.min_training_trades,
        minimum_lower_bound,
    )
    selected_holdout = measure(test, selection.threshold, confidence)
    wf = walk_forward(
        observations,
        thresholds,
        baseline,
        confidence,
        arguments.min_training_sessions,
        arguments.min_training_trades,
        minimum_lower_bound,
    )
    raw_candidates = []
    for threshold in thresholds:
        metrics = measure(train, threshold, confidence)
        if metrics.trades >= arguments.min_training_trades:
            raw_candidates.append(metrics)
    raw_best = max(raw_candidates, key=lambda result: result.net_total) if raw_candidates else None
    promotion = (
        selection.qualified
        and selected_holdout.trades >= arguments.min_holdout_trades
        and selected_holdout.net_total > 0.0
        and selected_holdout.lower_bound is not None
        and selected_holdout.lower_bound > minimum_lower_bound
    )

    print(
        f"reconstructed sessions={len(window.snapshots)} usable={len(observations)} "
        f"train={len(train)} holdout={len(test)}"
    )
    if skipped:
        print(f"skipped: {', '.join(f'{key}={value}' for key, value in skipped.items())}")
    print()
    print("threshold grid: train and untouched holdout")
    print(
        f"{'threshold':36} {'train n':>8} {'train net':>11} {'train LB':>11} "
        f"{'holdout n':>10} {'holdout net':>12} {'holdout LB':>12}"
    )
    print("-" * 108)
    for row in rows:
        threshold = Threshold(
            row["threshold"]["minimum_gap"],
            row["threshold"]["minimum_vwap_distance"],
        )
        train_metrics = row["train"]
        holdout_metrics = row["holdout"]
        print(
            f"{threshold.label():36} {train_metrics['trades']:8d} "
            f"{_money(train_metrics['net_total']):>11} "
            f"{_bound(train_metrics['lower_bound']):>11} "
            f"{holdout_metrics['trades']:10d} "
            f"{_money(holdout_metrics['net_total']):>12} "
            f"{_bound(holdout_metrics['lower_bound']):>12}"
        )

    print()
    print(f"selection: {selection.threshold.label()} ({selection.reason})")
    _print_metrics("selected training", selection.training_metrics, confidence)
    _print_metrics("selected holdout", selected_holdout, confidence)
    if raw_best is not None:
        print(f"best raw training total (not a promotion decision): {raw_best.threshold.label()}")
        _print_metrics("raw best training", raw_best, confidence)
    print()
    _print_metrics("walk-forward", wf.metrics, confidence)
    print(
        f"walk-forward threshold choices: warmup={wf.warmup_sessions} "
        f"fallback={wf.fallback_sessions} choices={wf.selected_thresholds}"
    )
    print()
    print(
        "PROMOTE: selected threshold cleared training and untouched holdout"
        if promotion
        else "DO NOT PROMOTE: no threshold has positive, stable out-of-sample evidence"
    )

    report = {
        "profile": PROFILE,
        "provenance": {
            "source": "reconstructed_option_prints_and_point_in_time_stock_bars",
            "days_requested": arguments.days,
            "sessions_reconstructed": len(window.snapshots),
            "sessions_used": len(observations),
            "modelled_relative_spread": relative_spread,
            "configured_baseline": baseline.as_dict(),
            "signal_floor": signal_threshold.as_dict(),
            "threshold_grid": [threshold.as_dict() for threshold in thresholds],
            "holdout_sessions": holdout,
            "confidence": confidence,
            "minimum_lower_bound": minimum_lower_bound,
            "minimum_scenario_sessions": arguments.min_scenario_sessions,
            "historical_liquidity_note": (
                "Historical option books and displayed sizes are unavailable. "
                "The replay applies the modelled spread but cannot prove live "
                "liquidity or depth gates."
            ),
        },
        "skipped": skipped,
        "grid": rows,
        "selection": {
            **selection.as_dict(),
            "holdout": selected_holdout.as_dict(),
            "promote": promotion,
        },
        "walk_forward": wf.as_dict(),
    }
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote research report to {arguments.json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
