"""The history the regime rule compares today's implied variance against.

The rule asks whether today's reading is high or low against its own recent
past. Until now the past it was handed was realised variance, taken from the
scenario set, because a recorded implied series did not exist. Implied variance
sits above realised on average, which is the variance risk premium, so a live
implied reading landed in the upper part of a realised distribution nearly
every time: across 550 rebuilt sessions the median reading sits at the 75.7th
percentile of the realised distribution, 489 of 550 read high_variance and one
reads low_variance. That is not a market call, it is a units mismatch, and it
left call_bwb and debit_vertical unselectable.

scripts/variance_history.py rebuilt the implied series off the same tape and
the same bisection the live path solves with, so the comparison can be implied
against implied. This reads that file for the live cycle.

What it refuses to do matters more than what it does:

  - a reading from the session being decided never enters its own history, so
    a rebuilt file that has caught up to today cannot leak into the call
  - a file whose newest session is older than the staleness budget raises,
    rather than quietly comparing today against a distribution from last month
  - a file for another symbol, or one carrying too few readings for a
    quantile, raises

Every one of those is a stand down rather than a guess. The rule is the only
thing deciding direction while no model ships, and a rule reading the wrong
history is worse than a rule that admits it cannot see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from convex.errors import DataError


@dataclass(frozen=True)
class VarianceHistory:
    """Prior implied variance readings, newest last, none from ``before``."""

    readings: tuple[float, ...]
    first_session: date
    last_session: date
    source: Path

    def __len__(self) -> int:
        return len(self.readings)

    def as_list(self) -> list[float]:
        return list(self.readings)


def load_history(
    path: Path, symbol: str, before: date, max_age_days: int, min_readings: int
) -> VarianceHistory:
    """Read the rebuilt implied series, or raise having explained why not."""
    if max_age_days <= 0:
        raise DataError(f"the staleness budget must be positive, found {max_age_days}")
    if min_readings <= 0:
        raise DataError(f"the minimum reading count must be positive, found {min_readings}")
    if not path.exists():
        raise DataError(
            f"no implied variance history at {path}. Rebuild it with "
            "scripts/variance_history.py before an entry runs"
        )

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise DataError(f"{path} is not readable as JSON: {error}") from error

    recorded = payload.get("symbol")
    if recorded != symbol:
        raise DataError(
            f"{path} holds a history for {recorded!r}, and this cycle trades {symbol!r}"
        )

    series = payload.get("series")
    if not isinstance(series, list) or not series:
        raise DataError(f"{path} carries no series to read")

    readings: list[tuple[date, float]] = []
    for row in series:
        try:
            session = date.fromisoformat(str(row["session"]))
            reading = float(row["iv_total"])
        except (KeyError, TypeError, ValueError) as error:
            raise DataError(f"{path} holds a row without a session and a reading: {row!r}") from error
        if reading <= 0.0:
            raise DataError(
                f"{path} holds a non-positive implied variance of {reading} on "
                f"{session.isoformat()}, which is not a variance"
            )
        # The session being decided is not part of its own history. A rebuild
        # that has caught up to today would otherwise hand the rule the answer.
        if session >= before:
            continue
        readings.append((session, reading))

    if len(readings) < min_readings:
        raise DataError(
            f"{path} holds {len(readings)} readings before {before.isoformat()}, and the "
            f"regime rule needs at least {min_readings}"
        )

    readings.sort()
    newest = readings[-1][0]
    age = (before - newest).days
    if age > max_age_days:
        raise DataError(
            f"the newest implied variance reading in {path} is from "
            f"{newest.isoformat()}, {age} days before {before.isoformat()}, which "
            f"exceeds the {max_age_days} day staleness budget. Rebuild it with "
            "scripts/variance_history.py"
        )

    return VarianceHistory(
        readings=tuple(reading for _, reading in readings),
        first_session=readings[0][0],
        last_session=newest,
        source=path,
    )
