"""Replaying the recorded sessions, gross and net.

The point of this module is one number reported twice. The research finding
that determines the whole design is that structures with a healthy gross Sharpe
can have a negative net Sharpe once realistic execution costs are paid, so
every statistic here is produced in both forms and shown side by side. A
backtest that reports only the gross figure is the thing that gets people into
these trades in the first place.

Three arms are measured on the same sessions:

  every session      trade the family every day it has a candidate. This is
                     the naive bot, and it is included because it is what the
                     comparison is against, not because it is a strategy
  classified         trade only when the family's own out-of-sample
                     probability clears one half. Hard mapping: full size or
                     nothing, never confidence-weighted
  basket             equal weight across whichever families fired that day,
                     which is the arrangement the research found beat every
                     single structure it tested

Every figure here reads a Sample's ``traded_`` triple, which is the candidate
the ranking crowned and the agent would have opened. The ``label_`` triple is
the median across the top few candidates and exists to teach the model; totting
it up would report the earnings of a trade nobody places.

Nothing here is annualised from a handful of sessions without saying so. With
four trading days a Sharpe ratio is a number, not evidence, and the report
carries the session count next to every figure so a reader can judge it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np

from convex.errors import DataError
from convex.structures.base import Family
from convex.training import Sample

TRADING_DAYS_PER_YEAR = 252

# Below this many observations an annualised Sharpe is arithmetic on noise. A
# handful of trades with similar results produces a ratio in the hundreds,
# which is not a good result. It is a small denominator. Refusing to print one
# is the honest behaviour, and over a four-session competition window it means
# no Sharpe is reported at all, which is correct.
MINIMUM_OBSERVATIONS_FOR_SHARPE = 20


@dataclass(frozen=True)
class Performance:
    """One arm's results, in both forms."""

    label: str
    sessions: int
    trades: int
    gross_total: float
    net_total: float
    cost_total: float
    gross_sharpe: float | None
    net_sharpe: float | None
    hit_rate: float | None
    expected_shortfall: float | None
    max_drawdown: float
    # The running total, one entry per trade in the order they were taken.
    # max_drawdown already walks this curve to find its worst fall; keeping it
    # is what lets the page draw the thing the drawdown is measured on, rather
    # than reporting a number about a shape nobody is shown.
    gross_curve: tuple[float, ...] = ()
    net_curve: tuple[float, ...] = ()

    @property
    def cost_share_of_gross(self) -> float | None:
        """How much of the gross result the execution consumed."""
        if self.gross_total <= 0.0:
            return None
        return self.cost_total / self.gross_total

    @property
    def survives_costs(self) -> bool:
        """Whether the arm still has an edge once it is paid for."""
        return self.net_sharpe is not None and self.net_sharpe > 0.0

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "sessions": self.sessions,
            "trades": self.trades,
            "gross_total": round(self.gross_total, 2),
            "net_total": round(self.net_total, 2),
            "cost_total": round(self.cost_total, 2),
            "gross_sharpe": None if self.gross_sharpe is None else round(self.gross_sharpe, 3),
            "net_sharpe": None if self.net_sharpe is None else round(self.net_sharpe, 3),
            "hit_rate": None if self.hit_rate is None else round(self.hit_rate, 4),
            "expected_shortfall": (
                None if self.expected_shortfall is None else round(self.expected_shortfall, 2)
            ),
            "max_drawdown": round(self.max_drawdown, 2),
            "cost_share_of_gross": (
                None if self.cost_share_of_gross is None else round(self.cost_share_of_gross, 4)
            ),
            "gross_curve": [round(value, 2) for value in self.gross_curve],
            "net_curve": [round(value, 2) for value in self.net_curve],
        }


def sharpe(series: np.ndarray) -> float | None:
    """Annualised Sharpe of a daily series, or nothing when it is undefined.

    A single session has no dispersion and a constant series has none either.
    Neither gets a number: dividing by a zero standard deviation would print an
    infinity that reads like an extraordinary result. Nor does a sample too
    small to support the statistic, for the same reason in slower motion.
    """
    if series.size < MINIMUM_OBSERVATIONS_FOR_SHARPE:
        return None
    deviation = float(series.std(ddof=1))
    if deviation <= 0.0:
        return None
    return float(series.mean() / deviation * np.sqrt(TRADING_DAYS_PER_YEAR))


def cumulative(series: np.ndarray) -> np.ndarray:
    """The running total of a per-trade series, which is its equity curve."""
    return np.cumsum(series)


def max_drawdown(series: np.ndarray) -> float:
    """Largest peak-to-trough fall of the cumulative result, as a positive."""
    if series.size == 0:
        return 0.0
    curve = cumulative(series)
    peak = np.maximum.accumulate(curve)
    return float(np.max(peak - curve))


def expected_shortfall(series: np.ndarray, confidence: float) -> float | None:
    """Mean of the worst tail, positive for a loss. None when too few sessions."""
    if not 0.0 < confidence < 1.0:
        raise DataError(f"ES confidence must lie strictly between 0 and 1, found {confidence}")
    if series.size == 0:
        return None
    count = max(1, int(np.floor(series.size * confidence)))
    return float(-np.sort(series)[:count].mean())


def measure(
    label: str, gross: Sequence[float], net: Sequence[float], costs: Sequence[float],
    sessions: int, confidence: float = 0.01,
) -> Performance:
    """Summarise one arm from its per-session results."""
    gross_array = np.asarray(list(gross), dtype=float)
    net_array = np.asarray(list(net), dtype=float)
    cost_array = np.asarray(list(costs), dtype=float)
    return Performance(
        label=label,
        sessions=sessions,
        trades=int(net_array.size),
        gross_total=float(gross_array.sum()),
        net_total=float(net_array.sum()),
        cost_total=float(cost_array.sum()),
        gross_sharpe=sharpe(gross_array),
        net_sharpe=sharpe(net_array),
        hit_rate=float((net_array > 0).mean()) if net_array.size else None,
        expected_shortfall=expected_shortfall(net_array, confidence),
        max_drawdown=max_drawdown(net_array),
        gross_curve=tuple(cumulative(gross_array).tolist()),
        net_curve=tuple(cumulative(net_array).tolist()),
    )


@dataclass
class BacktestReport:
    """Every arm, per family and for the basket."""

    per_family: dict[str, dict[str, Performance]] = field(default_factory=dict)
    basket: dict[str, Performance] = field(default_factory=dict)
    sessions: int = 0

    def as_dict(self) -> dict:
        return {
            "sessions": self.sessions,
            "per_family": {
                family: {arm: result.as_dict() for arm, result in arms.items()}
                for family, arms in self.per_family.items()
            },
            "basket": {arm: result.as_dict() for arm, result in self.basket.items()},
        }


def run(
    samples: Sequence[Sample],
    probabilities: dict[Family, dict[date, float]],
    confidence: float = 0.01,
) -> BacktestReport:
    """Replay the sessions with and without the classifier deciding.

    ``probabilities`` holds out-of-sample probabilities per family per session,
    produced by an expanding window. A session with no probability for a family
    is inside that family's burn-in and is left out of the classified arm
    rather than being traded on a model that did not exist yet.
    """
    if not samples:
        return BacktestReport()

    by_family: dict[Family, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_family[sample.family].append(sample)

    report = BacktestReport(sessions=len({sample.session_date for sample in samples}))
    basket_gross: dict[date, float] = defaultdict(float)
    basket_net: dict[date, float] = defaultdict(float)
    basket_cost: dict[date, float] = defaultdict(float)

    for family, rows in sorted(by_family.items(), key=lambda item: str(item[0])):
        rows.sort(key=lambda sample: sample.session_date)
        family_probabilities = probabilities.get(family, {})

        always = measure(
            "every session",
            [row.traded_gross_pnl for row in rows],
            [row.traded_net_pnl for row in rows],
            [row.traded_cost for row in rows],
            sessions=len(rows),
        )

        # Hard mapping: the probability decides whether, never how much.
        taken = [
            row for row in rows
            if family_probabilities.get(row.session_date, 0.0) > 0.5
        ]
        classified = measure(
            "classified",
            [row.traded_gross_pnl for row in taken],
            [row.traded_net_pnl for row in taken],
            [row.traded_cost for row in taken],
            sessions=len(rows),
        )
        report.per_family[str(family)] = {"every session": always, "classified": classified}

        for row in taken:
            basket_gross[row.session_date] += row.traded_gross_pnl
            basket_net[row.session_date] += row.traded_net_pnl
            basket_cost[row.session_date] += row.traded_cost

    days = sorted(basket_net)
    report.basket["classified"] = measure(
        "basket, classified",
        [basket_gross[day] for day in days],
        [basket_net[day] for day in days],
        [basket_cost[day] for day in days],
        sessions=report.sessions,
    )

    all_days = sorted({sample.session_date for sample in samples})
    every_gross: dict[date, float] = defaultdict(float)
    every_net: dict[date, float] = defaultdict(float)
    every_cost: dict[date, float] = defaultdict(float)
    for sample in samples:
        every_gross[sample.session_date] += sample.traded_gross_pnl
        every_net[sample.session_date] += sample.traded_net_pnl
        every_cost[sample.session_date] += sample.traded_cost
    report.basket["every session"] = measure(
        "basket, every session",
        [every_gross[day] for day in all_days],
        [every_net[day] for day in all_days],
        [every_cost[day] for day in all_days],
        sessions=report.sessions,
    )
    return report
