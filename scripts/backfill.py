"""Rebuild past sessions, label them, and fit the classifiers on them.

The classifier had nothing to learn from: training reads recorded chains and
the archive only holds sessions CONVEX has itself run. This walks backwards
through Alpaca's own calendar, rebuilds each session from the option tape (see
convex/reconstruct.py for what that can and cannot recover), labels each
family's choice with the live ranking, and fits the models.

Run it with:  .venv/bin/python -m scripts.backfill --days 400
Add --no-fit to build and report the dataset without writing models.

Two things this deliberately does not do. It does not write to the chain
archive, because a rebuilt session is not a session anyone observed. And it
does not pretend the modelled spread is a measurement: the figure used is
printed, carried into the report, and every sample records the cost it was
charged.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

from convex.agent import rank
from convex.classifier import (
    brier_score,
    calibration_slope,
    fit_family,
    save_models,
    walk_forward,
)
from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError, DataError
from convex.ledger import Action, Ledger, Record, new_cycle_id
from convex import reconstruct
from convex.reconstruct import as_snapshot, build as rebuild
from convex.structures.base import Family
from convex.scenarios import build as build_scenarios
from convex.training import (
    build_samples,
    reconstructed_feature_names_for,
    to_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=400, help="calendar days to walk back")
    parser.add_argument("--no-fit", action="store_true", help="build the dataset only")
    parser.add_argument(
        "--relative-spread",
        type=float,
        default=None,
        help="modelled relative spread; defaults to the configured liquidity threshold",
    )
    arguments = parser.parse_args()

    config = load()
    gateway = AlpacaGateway(config)
    ledger = Ledger(config.path_("paths.ledger"))
    zone = ZoneInfo(config.str_("session.timezone"))
    symbol = config.str_("underlying.symbol")

    spread = arguments.relative_spread
    if spread is None:
        spread = config.float_("liquidity.max_relative_spread")
        print(
            f"modelled relative spread {spread:.3f}, taken from the configured "
            "liquidity threshold. This is not a measurement of past spreads, "
            "which are gone; pass --relative-spread to override it."
        )

    now, _ = gateway.clock()
    end = now.date()
    start = end - timedelta(days=arguments.days)
    sessions = gateway.sessions(start, end)
    if not sessions:
        raise DataError(f"Alpaca's calendar lists no session between {start} and {end}")
    print(f"{len(sessions)} sessions on the calendar from {start} to {end}")

    bars = gateway.minute_bars(
        symbol,
        datetime.combine(start, dtime(0, 0), tzinfo=zone),
        datetime.combine(end, dtime(23, 59), tzinfo=zone),
    )
    print(f"{len(bars):,} underlying minute bars")

    entry_time = dtime.fromisoformat(config.str_("reconstruction.entry_time"))
    close_time = dtime.fromisoformat(config.str_("session.close_time"))
    minimum = config.float_("reconstruction.min_coverage")

    snapshots = []
    settlements: dict[date, float] = {}
    rebuilt_by_date: dict[date, object] = {}
    coverages: list[float] = []
    thin = 0
    for session in sessions:
        day = session.session_date
        entry_at = datetime.combine(day, entry_time, tzinfo=zone)
        close_at = datetime.combine(day, close_time, tzinfo=zone)
        before_entry = bars[bars.index <= entry_at]
        before_close = bars[bars.index <= close_at]
        if before_entry.empty or before_close.empty:
            continue
        spot_at_entry = float(before_entry.iloc[-1]["close"])
        spot_at_close = float(before_close.iloc[-1]["close"])

        rebuilt = rebuild(
            gateway, config, day, entry_at, close_at, spot_at_entry, spot_at_close
        )
        coverages.append(rebuilt.coverage)
        if rebuilt.coverage < minimum:
            thin += 1
            continue
        snapshots.append(as_snapshot(rebuilt, entry_at, spread))
        settlements[day] = spot_at_close
        rebuilt_by_date[day] = rebuilt

    print(
        f"rebuilt {len(snapshots)} usable sessions "
        f"(median ladder coverage {statistics.median(coverages):.2f}, "
        f"{thin} dropped below {minimum:.2f})"
    )

    def build_features(entries, spot, taken_at, close_at, prior_returns, family_pnl):
        """The rebuilt session's own predictors, not the live engine's.

        The live engine reads Greeks and the book off each chain row. A rebuilt
        row has neither, so the volatilities solved out of the tape are used
        instead and the four features that need the book are simply absent.
        """
        return reconstruct.features(
            rebuilt_by_date[taken_at.date()], taken_at, close_at, prior_returns, family_pnl
        )

    scenarios = build_scenarios(gateway, config)
    samples = build_samples(
        snapshots, settlements, scenarios, config, rank, build_features=build_features
    )
    print(f"{len(samples)} labelled samples across the families\n")

    by_family: dict[str, list] = {}
    for sample in samples:
        by_family.setdefault(str(sample.family), []).append(sample)

    report: dict[str, dict] = {}
    for family in sorted(by_family):
        rows = by_family[family]
        wins = sum(row.label for row in rows)
        gross = [row.gross_pnl for row in rows]
        net = [row.net_pnl for row in rows]
        eaten = sum(1 for row in rows if row.gross_pnl > 0.0 and row.net_pnl <= 0.0)
        report[family] = {
            "sessions": len(rows),
            "win_rate_net": round(wins / len(rows), 4),
            "mean_gross": round(statistics.fmean(gross), 2),
            "mean_net": round(statistics.fmean(net), 2),
            "profitable_gross_but_not_net": eaten,
        }
        print(
            f"  {family:<16} {len(rows):>4} sessions   "
            f"net win rate {wins / len(rows):.3f}   "
            f"mean gross {statistics.fmean(gross):>8.2f}   "
            f"mean net {statistics.fmean(net):>8.2f}   "
            f"cost turned {eaten} winners into losers"
        )

    if arguments.no_fit:
        print("\n--no-fit: nothing written")
        return 0

    models: dict[Family, object] = {}
    reports = []
    for name in sorted(by_family):
        family = Family(name)
        names = reconstructed_feature_names_for(family)
        matrix, labels = to_matrix(samples, family, names)

        # Out of sample first. Every session is predicted by a model fitted
        # only on the sessions before it, which is the only honest way to
        # report a hit rate on a strategy that will be run forward.
        probabilities, realised = walk_forward(family, matrix, labels, names, config)
        if probabilities.size:
            hits = float(((probabilities > 0.5).astype(int) == realised).mean())
            # The number that decides whether the hit rate means anything. When
            # a family wins one session in ten, predicting "loss" every day
            # scores ninety per cent, so a hit rate is only evidence of skill
            # insofar as it beats this.
            base_rate = float(max(realised.mean(), 1.0 - realised.mean()))
            fired = float((probabilities > 0.5).mean())
            # One standard error of the baseline on this many sessions. Skill
            # smaller than a couple of these is a handful of coin flips.
            sigma = math.sqrt(
                max(base_rate * (1.0 - base_rate), 1e-12) / probabilities.size
            )
            margin = config.float_("classifier.min_skill_sigmas") * sigma
            report[name].update(
                {
                    "out_of_sample_sessions": int(probabilities.size),
                    "hit_rate": round(hits, 4),
                    "majority_baseline": round(base_rate, 4),
                    "skill_over_baseline": round(hits - base_rate, 4),
                    "skill_required": round(margin, 4),
                    "share_of_sessions_traded": round(fired, 4),
                    "brier": round(brier_score(probabilities, realised), 4),
                    "calibration_slope": round(
                        calibration_slope(probabilities, realised), 4
                    ),
                }
            )
            print(
                f"  {name:<16} oos {probabilities.size:>4}   "
                f"hit {hits:.3f} vs baseline {base_rate:.3f} "
                f"({hits - base_rate:+.3f})   "
                f"Brier {brier_score(probabilities, realised):.4f}   "
                f"calib {calibration_slope(probabilities, realised):.3f}   "
                f"trades {fired:.1%}   needs {margin:+.3f}"
            )
        else:
            report[name]["out_of_sample_sessions"] = 0
            print(
                f"  {name:<16} never cleared the "
                f"{config.int_('classifier.min_train_days')}-session burn-in; "
                "the documented rule decides"
            )

        model, training_report = fit_family(family, matrix, labels, names, config)
        reports.append(training_report)

        # A model is only written if it beat the majority baseline out of
        # sample. Below it, the model has learned the base rate and nothing
        # else, and shipping it would put a fitted-looking thing in the live
        # path in place of the documented rule — which is worse than the rule,
        # because it looks like evidence. Standing down is the better outcome.
        skill = report[name].get("skill_over_baseline")
        needed = report[name].get("skill_required")
        if model is None:
            continue
        if skill is None or needed is None or skill < needed:
            print(
                f"  {name:<16} not written: {skill:+.3f} against the baseline does "
                f"not clear {needed:+.3f}, so the documented rule decides this family"
            )
            continue
        models[family] = model

    if models:
        save_models(models, reports, config.path_("paths.models"))
        print(f"\n{len(models)} models written to {config.path_('paths.models')}")

    ledger.append(
        Record(
            action=Action.CALIBRATION,
            cycle_id=new_cycle_id(),
            rationale=(
                f"Rebuilt {len(snapshots)} past sessions from the option tape and "
                f"labelled {len(samples)} samples net of a modelled relative spread "
                f"of {spread:.3f}. The book for those sessions is gone, so the "
                "liquidity and open-interest features are absent rather than zeroed "
                "and implied volatility is solved from prints."
            ),
            outcome={
                "reconstructed": True,
                "sessions": len(snapshots),
                "samples": len(samples),
                "modelled_relative_spread": spread,
                "families": report,
            },
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
