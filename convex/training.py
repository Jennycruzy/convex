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
    # Flow, which survives reconstruction where the book does not. tape_put_share
    # is the one to watch: it is the traded counterpart of implied skew, and skew
    # is the documented driver.
    "tape_volume",
    "tape_breadth",
    "tape_concentration",
    "tape_put_share",
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


# What a session rebuilt from the tape cannot supply, because the book that
# produced it is gone: the open-interest exposure proxy and the three liquidity
# terms. See convex/reconstruct.py. A model fitted on rebuilt history is fitted
# without these and says so; at 10:00 the live row carries them anyway and the
# model simply does not read them.
UNRECONSTRUCTABLE_FEATURES: tuple[str, ...] = (
    "gex_balance",
    "liq_half_spread",
    "liq_relative_spread",
    "liq_tightness",
)


def feature_names_for(family: Family) -> tuple[str, ...]:
    """The exact predictor row one family's model is fitted on."""
    return SHARED_FEATURES + tuple(f"{family}_{name}" for name in OWN_RESULT_FEATURES)


def reconstructed_feature_names_for(family: Family) -> tuple[str, ...]:
    """The same row, less the features a rebuilt session cannot honestly fill."""
    shared = tuple(n for n in SHARED_FEATURES if n not in UNRECONSTRUCTABLE_FEATURES)
    return shared + tuple(f"{family}_{name}" for name in OWN_RESULT_FEATURES)


@dataclass(frozen=True)
class Sample:
    """One training row, kept with enough context to be audited.

    Two sets of money, and keeping them apart is the whole point of the naming.

    The ``label_`` figures are what the model is taught to recognise. Under
    ``label_top_k`` above one they are the median across the top few ranked
    candidates, which is deliberate shrinkage against the winner's curse on a
    rebuilt chain. They describe a candidate nobody buys.

    The ``traded_`` figures are what the agent would have earned, from the one
    candidate the ranking actually crowns. Every statement about profit, in the
    replay, the write-up or the gate that decides whether a model ships, is
    about these and only these.

    They were one triple named ``net_pnl`` once. The shrinkage meant for the
    label reached the earnings that way, and the ship gate spent a run scoring
    a trade the agent would never place.
    """

    session_date: date
    family: Family
    features: dict[str, float]
    label: int
    label_gross_pnl: float
    label_cost: float
    label_net_pnl: float
    traded_gross_pnl: float
    traded_cost: float
    traded_net_pnl: float
    description: str

    @property
    def cost_share(self) -> float:
        """What fraction of the traded gross result the execution took.

        Above one is the finding the project is built on: a structure that made
        money before costs and lost after them. It reads the traded figures
        because it is a claim about the trade, not about the label.
        """
        if self.traded_gross_pnl <= 0.0:
            return float("inf")
        return self.traded_cost / self.traded_gross_pnl


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
    build_features=None,
    label_top_k: int = 1,
) -> list[Sample]:
    """Walk the recorded sessions in order and label each family's choice.

    ``rank`` is the live cycle's own ranking function, passed in rather than
    reimplemented, so the candidate labelled here is provably the candidate the
    agent would have opened.

    ``label_top_k`` is how many of the ranked candidates the label is taken
    across; one reproduces labelling the exact trade. See the note at the
    labelling step for why a rebuilt chain may want more than one.

    ``build_features`` defaults to the live feature engine. A rebuilt session
    has no Greeks and no book, so the backfill passes its own builder rather
    than fabricating them; the lagged per-family results still accumulate here,
    in session order, which is why this is a function and not a table of rows.
    """
    build_features = build_features or feature_engine.build
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
    # actually preceded each session. The whole set would include days after
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

        row = build_features(
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

            # Which of the ranked candidates the label is taken from.
            #
            # One is what the agent actually trades, and for a recorded chain it
            # is plainly the right answer: label the decision that was made. On a
            # rebuilt chain it is less obviously right. The ranking maximises net
            # edge over prices carrying reconstruction noise, so the candidate it
            # crowns is partly the one whose print is most wrong, and a label
            # attached to it measures that error as much as the strategy. Taking
            # the median outcome across the top few is the usual shrinkage
            # against that winner's curse.
            #
            # The trade is unchanged either way: the agent opens ordered[0]. This
            # only decides what the model is taught to recognise.
            top = ordered[: max(label_top_k, 1)]
            outcomes = [
                (
                    settlement_pnl_of(
                        entry.candidate, entry.estimate.profile.net_entry_debit, settlement
                    ),
                    entry.estimate.cost.total,
                )
                for entry in top
            ]
            nets = sorted(gross - cost for gross, cost in outcomes)
            net = nets[len(nets) // 2]
            gross = sorted(gross for gross, _ in outcomes)[len(outcomes) // 2]
            cost = sorted(cost for _, cost in outcomes)[len(outcomes) // 2]

            # What the agent would really have earned: the crowned candidate,
            # not the median of the few behind it. The family's own lagged
            # results follow this series, and so does every figure downstream
            # that claims to be money.
            traded_gross = settlement_pnl_of(
                best.candidate, best.estimate.profile.net_entry_debit, settlement
            )
            traded_cost = best.estimate.cost.total
            traded_net = traded_gross - traded_cost
            history.setdefault(str(family), []).append(traded_net)
            samples.append(
                Sample(
                    session_date=snapshot.session_date,
                    family=family,
                    features=dict(row.values),
                    label=1 if net > 0.0 else 0,
                    label_gross_pnl=round(gross, 2),
                    label_cost=round(cost, 2),
                    label_net_pnl=round(net, 2),
                    traded_gross_pnl=round(traded_gross, 2),
                    traded_cost=round(traded_cost, 2),
                    traded_net_pnl=round(traded_net, 2),
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
