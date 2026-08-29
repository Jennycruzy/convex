"""The terminal distribution the agent prices against.

An edge has to be measured under some distribution of where SPY finishes the
day. Using the risk-neutral density recovered from the chain would be circular:
under the market's own measure every structure prices to roughly zero edge by
construction, so nothing would ever look attractive and nothing would ever be
rejected for the right reason. The research finding this project is built on is
that 0DTE payoffs are driven by realised skewness, which is a statement about
the physical distribution, not the risk-neutral one.

So the scenario set is empirical: the actual 10:00-to-16:00 returns SPY printed
on every trading day in the lookback window, fetched from Alpaca's minute bars.
Each historical day becomes one equally weighted scenario. That makes expected
payoff, win rate and expected shortfall exact sums over real history rather than
draws from a fitted law, with no random seed anywhere in the pipeline: the same
inputs produce the same size, every time, which is what Law 11 requires.

The one adjustment offered is a volatility scaling, which stretches the
historical return set so its dispersion matches what today's chain implies.
Without it a quiet history would make today's structures look far safer than
the market believes they are. It is applied explicitly by the caller and
recorded on the scenario set, never applied silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from convex.config import Config
from convex.errors import DataError

# A session whose last bar before the entry time is older than this did not
# trade at the entry time, so it cannot contribute an entry-to-close return.
_MAX_ENTRY_BAR_STALENESS_MINUTES = 5


@dataclass(frozen=True)
class ScenarioSet:
    """Equally weighted terminal log returns from entry time to the close."""

    log_returns: np.ndarray
    source_days: tuple[date, ...]
    entry_time: time
    exit_time: time
    volatility_scale: float
    built_at: datetime

    def __post_init__(self) -> None:
        if self.log_returns.ndim != 1 or self.log_returns.size == 0:
            raise DataError("a scenario set needs a non-empty one-dimensional return array")
        if len(self.source_days) != self.log_returns.size:
            raise DataError(
                f"{self.log_returns.size} returns but {len(self.source_days)} source days"
            )
        if not np.all(np.isfinite(self.log_returns)):
            raise DataError("scenario returns contain a non-finite value")

    def __len__(self) -> int:
        return int(self.log_returns.size)

    @property
    def realised_sigma(self) -> float:
        """Standard deviation of the session return across the scenario set."""
        return float(np.std(self.log_returns, ddof=1))

    @property
    def realised_skew(self) -> float:
        """Sample skewness, the moment the research identifies as the driver."""
        centred = self.log_returns - self.log_returns.mean()
        sigma = np.std(self.log_returns)
        if sigma <= 0.0:
            raise DataError("scenario returns have zero dispersion")
        return float(np.mean(centred**3) / sigma**3)

    def annualised_variance(self, tau: float) -> np.ndarray:
        """Each session's realised variance, annualised to match implied units.

        The regime rule compares today's implied variance against a history of
        variance readings. Recorded implied variance is the natural comparison
        but does not exist until the agent has run for weeks, so the comparison
        is made against realised session variance over the same clock, scaled
        by the same tau the implied figure uses. The gap between the two is the
        variance risk premium, which is exactly what a regime call is about.
        """
        if tau <= 0.0:
            raise DataError(f"tau must be positive to annualise variance, found {tau}")
        return (self.log_returns**2) / tau

    def prices(self, spot: float) -> np.ndarray:
        """Terminal underlying prices implied by each scenario."""
        if spot <= 0.0:
            raise DataError(f"spot must be positive, found {spot}")
        return spot * np.exp(self.log_returns)

    def scaled_to(self, target_sigma: float) -> "ScenarioSet":
        """Return a copy whose dispersion matches a target session sigma."""
        if target_sigma <= 0.0:
            raise DataError(f"target sigma must be positive, found {target_sigma}")
        current = self.realised_sigma
        if current <= 0.0:
            raise DataError("cannot rescale a scenario set with zero dispersion")
        factor = target_sigma / current
        mean = float(self.log_returns.mean())
        return ScenarioSet(
            log_returns=mean + (self.log_returns - mean) * factor,
            source_days=self.source_days,
            entry_time=self.entry_time,
            exit_time=self.exit_time,
            volatility_scale=self.volatility_scale * factor,
            built_at=self.built_at,
        )

    def describe(self) -> dict[str, float | int | str]:
        return {
            "scenarios": len(self),
            "first_day": self.source_days[0].isoformat(),
            "last_day": self.source_days[-1].isoformat(),
            "sigma": round(self.realised_sigma, 6),
            "skew": round(self.realised_skew, 4),
            "volatility_scale": round(self.volatility_scale, 4),
        }


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def build_from_bars(bars: pd.DataFrame, config: Config) -> ScenarioSet:
    """Turn Alpaca minute bars into one session return per trading day.

    The entry price is the close of the minute bar that covers the entry time,
    and the exit price is the close of the last bar of the regular session. A
    day missing either bar is dropped and counted, never patched.
    """
    zone = ZoneInfo(config.str_("session.timezone"))
    entry_time = _parse_time(config.str_("session.entry_time"))
    exit_time = _parse_time(config.str_("session.close_time"))

    frame = bars.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.droplevel("symbol")
    if "close" not in frame.columns:
        raise DataError(f"minute bars are missing a close column: {list(frame.columns)}")

    local = frame.tz_convert(zone) if frame.index.tz is not None else frame.tz_localize("UTC").tz_convert(zone)
    local = local.sort_index()

    returns: list[float] = []
    days: list[date] = []
    skipped: list[date] = []
    for session_day, day_bars in local.groupby(local.index.date):
        entry_bars = day_bars[day_bars.index.time <= entry_time]
        exit_bars = day_bars[day_bars.index.time <= exit_time]
        if entry_bars.empty or exit_bars.empty:
            skipped.append(session_day)
            continue
        entry_price = float(entry_bars["close"].iloc[-1])
        exit_price = float(exit_bars["close"].iloc[-1])
        if entry_price <= 0.0 or exit_price <= 0.0:
            skipped.append(session_day)
            continue
        entry_stamp = entry_bars.index[-1]
        entry_target = pd.Timestamp(datetime.combine(session_day, entry_time), tz=zone)
        if entry_target - entry_stamp > pd.Timedelta(minutes=_MAX_ENTRY_BAR_STALENESS_MINUTES):
            # The last bar at or before the entry time is stale by more than the
            # tolerance, so this session did not actually trade at the entry
            # time and is dropped rather than approximated.
            skipped.append(session_day)
            continue
        returns.append(float(np.log(exit_price / entry_price)))
        days.append(session_day)

    if not returns:
        raise DataError(
            f"no usable sessions in the bar history; {len(skipped)} days lacked "
            f"a bar at {entry_time} or at {exit_time}"
        )
    return ScenarioSet(
        log_returns=np.asarray(returns, dtype=float),
        source_days=tuple(days),
        entry_time=entry_time,
        exit_time=exit_time,
        volatility_scale=1.0,
        built_at=datetime.now(tz=zone),
    )


def build(gateway, config: Config, asof: date | None = None) -> ScenarioSet:
    """Fetch the lookback window from Alpaca and build the scenario set."""
    symbol = config.str_("underlying.symbol")
    lookback = config.int_("data.scenario_lookback_days")
    end = datetime.combine(asof or date.today(), time(23, 59), tzinfo=ZoneInfo("UTC"))
    start = end - timedelta(days=lookback)
    bars = gateway.minute_bars(symbol, start, end)
    return build_from_bars(bars, config)


def save(scenarios: ScenarioSet, directory: Path) -> Path:
    """Archive a scenario set so a decision can be reproduced exactly."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = scenarios.built_at.strftime("%Y%m%dT%H%M%S")
    path = directory / f"scenarios-{stamp}.npz"
    np.savez_compressed(
        path,
        log_returns=scenarios.log_returns,
        source_days=np.array([day.isoformat() for day in scenarios.source_days]),
    )
    path.with_suffix(".json").write_text(json.dumps(scenarios.describe(), indent=2))
    return path
