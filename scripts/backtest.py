"""Replay the recorded sessions and report gross against net.

This is the project's own version of the table it is built on. The research
found iron butterfly and condor structures at a gross Sharpe of 0.77 and a net
of −0.20; this runs the same comparison on the structures CONVEX actually
trades, on the chains CONVEX actually recorded, and prints both columns for
every arm so the gap is visible rather than asserted.

Read the session count before reading the Sharpe. Over a handful of sessions a
Sharpe ratio is arithmetic, not evidence, and this prints the count next to
every figure precisely so nobody quotes one without the other.

Run it with:  .venv/bin/python -m scripts.backtest
              .venv/bin/python -m scripts.backtest --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from zoneinfo import ZoneInfo

from convex import archive, backtest, training
from convex.agent import rank
from convex.classifier import fit_family
from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError
from convex.scenarios import build as build_scenarios
from convex.structures.base import Family
from scripts.train import settlements


def out_of_sample_probabilities(samples, config) -> dict:
    """Expanding-window probability per family per session.

    Session t is scored by a model fitted only on that family's sessions before
    t. Sessions inside the burn-in get no probability at all, so the classified
    arm cannot trade on a model that did not exist yet.
    """
    minimum = config.int_("classifier.min_train_days")
    probabilities: dict = {}
    for name in config.list_("structures.enabled"):
        family = Family(str(name))
        rows = sorted(
            (s for s in samples if s.family is family), key=lambda s: s.session_date
        )
        if len(rows) <= minimum:
            continue
        matrix, labels = training.to_matrix(rows, family)
        names = training.feature_names_for(family)
        per_day: dict = {}
        for index in range(minimum, matrix.shape[0]):
            model, _ = fit_family(family, matrix[:index], labels[:index], names, config)
            if model is None:
                continue
            per_day[rows[index].session_date] = model.probability(matrix[index])
        if per_day:
            probabilities[family] = per_day
    return probabilities


def show(value, places: int = 3, width: int = 8) -> str:
    return f"{value:{width}.{places}f}" if value is not None else " " * (width - 1) + "—"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="also write the full report to this path")
    arguments = parser.parse_args()

    config = load()
    symbol = config.str_("underlying.symbol")
    zone = ZoneInfo(config.str_("session.timezone"))
    chains = config.path_("paths.chain_archive")

    recorded = archive.sessions(chains)
    if not recorded:
        print(
            "no recorded chains to replay. The agent archives each 10:00 snapshot "
            "as it runs; there are none yet."
        )
        return 1

    gateway = AlpacaGateway(config)
    closes = settlements(gateway, symbol, recorded, zone)
    scenarios = build_scenarios(gateway, config)
    samples = training.build_samples(
        list(archive.read_all(chains)), closes, scenarios, config, rank
    )
    if not samples:
        print("no session could be labelled; nothing to replay")
        return 1

    probabilities = out_of_sample_probabilities(samples, config)
    report = backtest.run(samples, probabilities, config.float_("risk.es_confidence"))

    sessions = report.sessions
    print(f"\nreplayed {sessions} recorded session(s)\n")
    if sessions < 30:
        print(
            f"  Note: {sessions} sessions is far too few for a Sharpe ratio to mean "
            "anything.\n  The figures below are arithmetic on a small sample and are "
            "reported as such.\n"
        )

    header = f"{'':<26} {'trades':>6} {'gross SR':>9} {'net SR':>9} {'gross $':>10} {'net $':>10} {'cost $':>9}"
    print(header)
    print("-" * len(header))

    def line(label: str, arm) -> None:
        print(
            f"{label:<26} {arm.trades:6d} {show(arm.gross_sharpe, 2, 9)} "
            f"{show(arm.net_sharpe, 2, 9)} {arm.gross_total:10,.2f} "
            f"{arm.net_total:10,.2f} {arm.cost_total:9,.2f}"
        )

    for family, arms in sorted(report.per_family.items()):
        for name, arm in arms.items():
            line(f"{family} · {name}", arm)
        print()

    for name, arm in sorted(report.basket.items()):
        line(arm.label, arm)

    print()
    for name, arm in sorted(report.basket.items()):
        if arm.gross_sharpe is not None and arm.net_sharpe is not None:
            verdict = "survives its costs" if arm.survives_costs else "does NOT survive its costs"
            print(f"  {arm.label}: gross {arm.gross_sharpe:.2f} → net {arm.net_sharpe:.2f}, {verdict}")

    if arguments.json:
        payload = report.as_dict()
        payload["note"] = (
            "Replayed on chains recorded by the agent at 10:00 ET. Sharpe figures over "
            f"{sessions} sessions are arithmetic, not evidence."
        )
        with open(arguments.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote the full report to {arguments.json}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
