"""Turning recorded chains into something a classifier can learn from.

One row per session per family. The predictors are the same 10:00 feature row
the agent computes live, recomputed from the recorded chain. The label is
whether the candidate the agent *would have chosen* that day finished the
session profitable after its own execution cost.

Three things about that definition matter.

**The label is net, not gross.** A structure that made money before costs and
lost after them is labelled a loss, because that is what it was. Training on
gross labels would teach the model to pick exactly the trades this project
exists to refuse.

**The candidate is the one the agent would have picked**, chosen by the same
ranking the live cycle uses, not the best candidate in hindsight. A model
trained on the best available strike learns to predict a decision nobody makes.

**Nothing looks forward.** A session's features come from its own 10:00
snapshot, and the lagged per-family results come only from sessions strictly
before it, accumulated in order as the loop walks forward.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np

from convex import features as feature_engine
from convex.archive import ChainSnapshot
from convex.config import Config
from convex.costs import CostModel
from convex.edge import evaluate
from convex.errors import DataError
from convex.scenarios import ScenarioSet
from convex.structures import build_candidates
from convex.structures.base import Candidate, Family

# Predictors every family sees, in a fixed order so a saved model and a live
# feature row can never drift apart. Implied skew leads because it is the
# highest-value feature in the research: realized skewness, not variance,
# is what drives 0DTE results.
SHARED_FEATURES: tuple[str, ...] = (
    "implied_skew",
    "iv_total",
    "iv_up",
    "iv_dn",
    "slope_up",
    "slope_dn",
    "gex_balance",
    "rv_lag1",
    "ret_lag1",
    "rv_5",
    "rskew",
    "liq_half_spread",
    "liq_relative_spread",
    "liq_tightness",
)

# Each family also sees its own past results and nobody else's. The research
# uses lagged strategy PnL per strategy; feeding one family another family's
# history would be a different model than the one being cited.
OWN_RESULT_FEATURES: tuple[str, ...] = ("pnl_lag1", "pnl_mean5", "pnl_std5")

# The realised-moment features are undefined below this many prior sessions.
_MINIMUM_PRIOR_SESSIONS = 5


def feature_names_for(family: Family) -> tuple[str, ...]:
    """The exact predictor row one family's model is fitted on."""
    return SHARED_FEATURES + tuple(f"{family}_{name}" for name in OWN_RESULT_FEATURES)


@dataclass(frozen=True)
class Sample:
    """One training row, kept with enough context to be audited."""

    session_date: date
    family: Family
    features: dict[str, float]
    label: int
    gross_pnl: float
    cost: float
    net_pnl: float
    description: str

    @property
    def cost_share(self) -> float:
        """What fraction of a positive gross result the execution took.

        Above one is the finding the project is built on: a structure that made
        money before costs and lost after them.
        """
        return self.cost / self.gross_pnl if self.gross_pnl > 0.0 else float("inf")


def settlement_pnl_of(
    candidate: Candidate, net_entry_debit: float, settlement_price: float
) -> float:
    """Payoff of one lot at expiry, in dollars, before execution cost."""
    multiplier = candidate.legs[0].contract.multiplier
    intrinsic = sum(
        leg.ratio * leg.contract.intrinsic(settlement_price) for leg in candidate.legs
    )
    return (intrinsic - net_entry_debit) * multiplier


def build_samples(
    snapshots: Sequence[ChainSnapshot],
    settlements: dict[date, float],
    scenarios: ScenarioSet,
    config: Config,
    rank,
) -> list[Sample]:
    """Walk the recorded sessions in order and label each family's choice.

    ``rank`` is the live cycle's own ranking function, passed in rather than
    reimplemented, so the candidate labelled here is provably the candidate the
    agent would have opened.
    """
    if not snapshots:
        return []
    cost_model = CostModel.from_config(config)
    tie_break = [str(name) for name in config.list_("structures.tie_break_order")]
    confidence = config.float_("risk.es_confidence")

    # Every enabled family is seeded with an empty history so its lagged
    # columns exist from the first session onward. Without this the first day
    # produces a narrower feature row than the second, which is a shape bug
    # that would only surface once a model was already fitted.
    history: dict[str, list[float]] = {
        str(name): [] for name in config.list_("structures.enabled")
    }
    samples: list[Sample] = []

    # The realised-moment features need sessions before the one being labelled.
    # They come from the scenario set's own history, sliced to the days that
    # actually preceded each session — the whole set would include days after
    # it, which is look-ahead wearing the costume of a longer sample.
    scenario_returns = list(zip(scenarios.source_days, scenarios.log_returns.tolist()))
    realised_after: list[tuple[date, float]] = []

    for snapshot in sorted(snapshots, key=lambda item: item.session_date):
        settlement = settlements.get(snapshot.session_date)
        if settlement is None:
            # A session whose close is unknown cannot be labelled. It is
            # skipped and counted, never labelled with a guess.
            continue

        close_at = snapshot.taken_at.replace(
            hour=16, minute=0, second=0, microsecond=0
        )
        if close_at <= snapshot.taken_at:
            raise DataError(
                f"{snapshot.session_date}: the snapshot at {snapshot.taken_at} is not "
                "before the session close"
            )
        prior_returns = [
            value
            for day, value in scenario_returns + realised_after
            if day < snapshot.session_date
        ]
        if len(prior_returns) < _MINIMUM_PRIOR_SESSIONS:
            # Not enough history behind this session to compute its lagged
            # moments. It is dropped and counted, never padded with zeros.
            continue

        row = feature_engine.build(
            snapshot.entries,
            snapshot.spot,
            snapshot.taken_at,
            close_at,
            prior_returns,
            {name: list(values) for name, values in history.items()},
        )

        for family, candidates in build_candidates(
            snapshot.entries, config, snapshot.spot
        ).items():
            priced = [
                (
                    candidate,
                    evaluate(
                        candidate.legs, scenarios, cost_model, snapshot.spot, 1, confidence
                    ),
                )
                for candidate in candidates
            ]
            if not priced:
                continue
            ordered = rank(
                [_Priced(candidate, estimate) for candidate, estimate in priced], tie_break
            )
            best = ordered[0]
            gross = settlement_pnl_of(
                best.candidate, best.estimate.profile.net_entry_debit, settlement
            )
            net = gross - best.estimate.cost.total
            history.setdefault(str(family), []).append(net)
            samples.append(
                Sample(
                    session_date=snapshot.session_date,
                    family=family,
                    features=dict(row.values),
                    label=1 if net > 0.0 else 0,
                    gross_pnl=round(gross, 2),
                    cost=round(best.estimate.cost.total, 2),
                    net_pnl=round(net, 2),
                    description=best.candidate.description,
                )
            )

        realised_after.append(
            (snapshot.session_date, float(np.log(settlement / snapshot.spot)))
        )

    return samples


class _Priced:
    """The shape the live ranking function expects, without importing it."""

    __slots__ = ("candidate", "estimate")

    def __init__(self, candidate, estimate) -> None:
        self.candidate = candidate
        self.estimate = estimate


def to_matrix(
    samples: Sequence[Sample], family: Family, names: Sequence[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """One family's rows as a matrix and a label vector, in session order."""
    names = tuple(names) if names is not None else feature_names_for(family)
    rows = [sample for sample in samples if sample.family is family]
    if not rows:
        return np.empty((0, len(names))), np.empty(0, dtype=int)
    missing = {
        name for name in names for sample in rows if name not in sample.features
    }
    if missing:
        raise DataError(f"{family}: recorded features are missing {sorted(missing)}")
    matrix = np.array(
        [[sample.features[name] for name in names] for sample in rows], dtype=float
    )
    labels = np.array([sample.label for sample in rows], dtype=int)
    return matrix, labels
