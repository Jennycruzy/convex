"""Isolated, dry-run profiles for the final paper-trading tournament.

This module deliberately contains no broker write. ``scripts.tournament``
wraps the real gateway in ``DryRunGateway`` and each profile writes evidence to
its own ledger. A profile is therefore comparable without being able to alter
the competition account by accident.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from convex.config import Config
from convex.errors import DataError
from convex.instruments import Right
from convex.structures.base import Candidate, Family


@dataclass(frozen=True)
class Profile:
    """A named candidate universe and execution observation policy."""

    name: str
    label: str
    families: tuple[Family, ...]
    description: str
    fill_ladder_ticks: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrendSignal:
    """A fully observed intraday direction, or a justified refusal."""

    direction: str
    last_close: float | None
    vwap: float | None
    opening_high: float | None
    opening_low: float | None
    detail: str

    @property
    def tradeable(self) -> bool:
        return self.direction in {"bullish", "bearish"}


def profiles(config: Config) -> tuple[Profile, ...]:
    """Profiles declared in YAML; unknown structure names fail loudly."""
    raw = config.get("tournament.profiles")
    if not isinstance(raw, dict):
        raise DataError("tournament.profiles must be a mapping")
    result: list[Profile] = []
    for name, values in raw.items():
        if not isinstance(values, dict):
            raise DataError(f"tournament profile {name!r} must be a mapping")
        families = values.get("families")
        if not isinstance(families, list) or not families:
            raise DataError(f"tournament profile {name!r} must name at least one family")
        ticks = values.get("fill_ladder_ticks", [])
        if not isinstance(ticks, list) or any(not isinstance(tick, int) or tick < 0 for tick in ticks):
            raise DataError(f"tournament profile {name!r} has invalid fill_ladder_ticks")
        result.append(Profile(
            name=str(name), label=str(values.get("label", name)),
            families=tuple(Family(str(family)) for family in families),
            description=str(values.get("description", "")),
            fill_ladder_ticks=tuple(ticks),
        ))
    return tuple(result)


def profile_config(config: Config, profile: Profile) -> Config:
    """Give one profile its family universe and append-only receipt file."""
    names = [str(family) for family in profile.families]
    return config.with_overrides({
        "structures.enabled": names,
        "structures.tie_break_order": names,
        "paths.ledger": config.str_(f"tournament.profiles.{profile.name}.ledger"),
    })


def submission_config(config: Config, profile: Profile) -> Config:
    """Prepare the one profile permitted to send a competition paper order.

    A submitted position must use the canonical ledger. The manager and
    reconciler deliberately read that ledger, and an isolated receipt would
    create a position the settlement guard could not attribute. One concurrent
    structure is enforced even though the base strategy normally allows four.
    """
    if profile.name not in {"skew_bwb", "execution_bwb"}:
        raise DataError(f"{profile.name} is observation-only and cannot be submitted")
    names = [str(family) for family in profile.families]
    return config.with_overrides({
        "structures.enabled": names,
        "structures.tie_break_order": names,
        "risk.max_concurrent_structures": 1,
    })


def intraday_trend(gateway, config: Config, now: datetime) -> TrendSignal:
    """Require both a 15-minute range break and VWAP confirmation.

    This is intentionally a gate, not a forecast. A vertical profile receives
    no direction when either observable disagrees. Bars are read from Alpaca;
    no value is carried forward from an earlier tournament pass.
    """
    zone = ZoneInfo(config.str_("session.timezone"))
    local = now.astimezone(zone)
    opening = datetime.combine(local.date(), time(9, 30), tzinfo=zone)
    if local <= opening:
        return TrendSignal("flat", None, None, None, None, "market has not opened")
    bars = gateway.minute_bars(config.str_("underlying.symbol"), opening, now)
    if bars.empty or "close" not in bars or "high" not in bars or "low" not in bars:
        raise DataError("trend profile needs non-empty minute bars with high, low, and close")
    local_bars = bars.tz_convert(zone)
    minutes = config.int_("tournament.trend.opening_range_minutes")
    opening_bars = local_bars.iloc[:minutes]
    if len(opening_bars) < minutes:
        return TrendSignal("flat", None, None, None, None, "opening range is incomplete")
    if "volume" not in local_bars or float(local_bars["volume"].sum()) <= 0.0:
        raise DataError("trend profile needs positive displayed minute-bar volume")
    last = float(local_bars["close"].iloc[-1])
    vwap = float((local_bars["close"] * local_bars["volume"]).sum() / local_bars["volume"].sum())
    high = float(opening_bars["high"].max())
    low = float(opening_bars["low"].min())
    distance = config.float_("tournament.trend.min_vwap_distance_pct")
    if last > high and last >= vwap * (1.0 + distance):
        direction = "bullish"
    elif last < low and last <= vwap * (1.0 - distance):
        direction = "bearish"
    else:
        direction = "flat"
    return TrendSignal(direction, last, vwap, high, low,
                       f"{direction}: close {last:.2f}, VWAP {vwap:.2f}, opening range {low:.2f}–{high:.2f}")


def trend_candidate_filter(signal: TrendSignal) -> Callable[[Candidate], bool]:
    """Keep only the vertical direction that the observed trend permits."""
    if signal.direction == "bullish":
        wanted = Right.CALL
    elif signal.direction == "bearish":
        wanted = Right.PUT
    else:
        return lambda candidate: False
    return lambda candidate: all(leg.contract.right is wanted for leg in candidate.legs)


def fill_ladder(limit_price: float, ticks: Iterable[int], tick_size: float) -> list[float]:
    """Describe, never submit, increasingly marketable limits.

    Positive debit limits worsen by a tick; negative credit limits become less
    negative. The runner records this plan on a dry-run receipt only. An
    authorized execution implementation must re-price net edge before every
    rung rather than treating this observation plan as permission to cross.
    """
    return [round(limit_price + tick * tick_size, 2) for tick in ticks]
