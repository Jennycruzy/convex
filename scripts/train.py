"""Fit the per-family classifiers on the chains the agent recorded.

The research protocol, run on our own data: an expanding window, predictors
standardised on the training window alone, and out-of-sample probabilities from
models that only ever saw sessions strictly earlier than the one they predict.
Hit rate, Brier score and calibration slope are reported per family, not
accuracy on its own, which on an unbalanced label says almost nothing.

This trains on recorded chains, never on simulated ones. That is a hard limit
rather than a preference: an expired contract's book cannot be fetched back, so
a session the agent did not record is a session that can never be labelled.
Expect this to report too little history early on. Saying so is the correct
outcome, and the agent runs the documented volatility-regime rule instead.

Run it with:  .venv/bin/python -m scripts.train
              .venv/bin/python -m scripts.train --dry-run   (report, save nothing)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from convex import archive, training
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
from convex.errors import ConvexError
from convex.scenarios import build as build_scenarios
from convex.structures.base import Family


def settlements(gateway: AlpacaGateway, symbol: str, days: list, zone: ZoneInfo) -> dict:
    """Each recorded session's official close, read from the market data API.

    The label depends on exactly one number per session and this is it, so it
    is fetched rather than remembered: the last quote the agent happened to see
    at 15:59 is not where the session closed.
    """
    if not days:
        return {}
    start = datetime.combine(min(days), time(9, 0), tzinfo=zone)
    end = datetime.combine(max(days), time(16, 5), tzinfo=zone)
    bars = gateway.minute_bars(symbol, start, end)
    local = bars.tz_convert(zone)

    closes: dict = {}
    for day in days:
        session_close = datetime.combine(day, time(16, 0), tzinfo=zone)
        within = local[(local.index <= session_close) & (local.index.date == day)]
        if within.empty:
            continue
        closes[day] = float(within["close"].iloc[-1])
    return closes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report but do not save")
    arguments = parser.parse_args()

    config = load()
    symbol = config.str_("underlying.symbol")
    zone = ZoneInfo(config.str_("session.timezone"))
    chains = config.path_("paths.chain_archive")

    recorded = archive.sessions(chains)
    print(f"recorded sessions: {len(recorded)}" + (f" ({recorded[0]} to {recorded[-1]})" if recorded else ""))
    if not recorded:
        print(
            "\nnothing to train on. The agent records each 10:00 chain as it runs, "
            "and training reads those recordings; there are none yet.\n"
            "Until there are, every family uses the documented volatility-regime "
            "rule and each ledger record says so."
        )
        return 1

    gateway = AlpacaGateway(config)
    closes = settlements(gateway, symbol, recorded, zone)
    print(f"settlement prices found for {len(closes)} of {len(recorded)} sessions")

    scenarios = build_scenarios(gateway, config)
    print(f"scenario set: {scenarios.describe()}")

    snapshots = list(archive.read_all(chains))
    samples = training.build_samples(snapshots, closes, scenarios, config, rank)
    print(f"labelled {len(samples)} rows across {len({s.session_date for s in samples})} sessions\n")

    minimum = config.int_("classifier.min_train_days")
    models = {}
    reports = []
    print(f"{'family':<16} {'rows':>5} {'wins':>5} {'hit':>7} {'brier':>7} {'calib':>7}  note")
    for name in config.list_("structures.enabled"):
        family = Family(str(name))
        matrix, labels = training.to_matrix(samples, family)
        names = training.feature_names_for(family)
        model, report = fit_family(family, matrix, labels, names, config)
        reports.append(report)

        hit = brier = calib = None
        if matrix.shape[0] > minimum:
            # Out-of-sample: every probability comes from a model that never
            # saw the session it is predicting.
            probabilities, realised, _ = walk_forward(
                family, matrix, labels, names, config
            )
            if probabilities.size:
                hit = float(((probabilities > 0.5).astype(int) == realised).mean())
                brier = brier_score(probabilities, realised)
                calib = calibration_slope(probabilities, realised)

        def show(value):
            return f"{value:7.3f}" if value is not None else "      ·"

        print(
            f"{str(family):<16} {report.samples:5d} {report.positives:5d} "
            f"{show(hit)} {show(brier)} {show(calib)}  {report.note}"
        )
        if model is not None:
            models[family] = model

    if not models:
        print(
            f"\nNo family cleared the {minimum}-session burn-in, so nothing was fitted. "
            "This is reported rather than worked around: a model fitted on thirty rows "
            "and presented as if it meant something is worse than the documented rule."
        )
        return 1

    if arguments.dry_run:
        print(f"\ndry run: {len(models)} model(s) fitted, nothing written")
        return 0

    path = save_models(models, reports, config.path_("paths.models"))
    print(f"\nwrote {len(models)} model(s) and their reports to {path}")
    print("the next cycle will use them, and each record will say so")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
