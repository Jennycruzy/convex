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
import pathlib
import sys
from zoneinfo import ZoneInfo

from convex import archive, backtest, reconstruct, training
from convex.agent import rank
from convex.classifier import fit_family
from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError
from convex.scenarios import build as build_scenarios
from convex.structures.base import Family
from scripts.train import settlements


def out_of_sample_probabilities(samples, config, names_for=None) -> dict:
    """Expanding-window probability per family per session.

    Session t is scored by a model fitted only on that family's sessions before
    t. Sessions inside the burn-in get no probability at all, so the classified
    arm cannot trade on a model that did not exist yet.
    """
    minimum = config.int_("classifier.min_train_days")
    names_for = names_for or training.feature_names_for
    probabilities: dict = {}
    for name in config.list_("structures.enabled"):
        family = Family(str(name))
        rows = sorted(
            (s for s in samples if s.family is family), key=lambda s: s.session_date
        )
        if len(rows) <= minimum:
            continue
        names = names_for(family)
        matrix, labels = training.to_matrix(rows, family, names)
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
    return f"{value:{width}.{places}f}" if value is not None else " " * (width - 1) + "·"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstructed",
        action="store_true",
        help="replay sessions rebuilt from the option tape instead of recorded chains",
    )
    parser.add_argument(
        "--days", type=int, default=400, help="with --reconstructed, days to walk back"
    )
    parser.add_argument(
        "--relative-spread",
        type=float,
        default=0.05,
        help="with --reconstructed, the modelled relative spread per leg",
    )
    parser.add_argument(
        "--label-top-k", type=int, default=1,
        help="label across the top k ranked candidates (see convex/training.py)",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="",
        help="write the full report; with no path it goes where the dashboard reads it",
    )
    arguments = parser.parse_args()

    config = load()
    symbol = config.str_("underlying.symbol")
    zone = ZoneInfo(config.str_("session.timezone"))
    chains = config.path_("paths.chain_archive")

    gateway = AlpacaGateway(config)
    scenarios = build_scenarios(gateway, config)
    names_for = training.feature_names_for
    provenance = "recorded"

    if arguments.reconstructed:
        window = reconstruct.rebuild_window(
            gateway, config, arguments.days, arguments.relative_spread
        )
        if not window.snapshots:
            print("no session could be rebuilt; nothing to replay")
            return 1
        print(
            f"rebuilt {len(window.snapshots)} sessions from the option tape, "
            f"median ladder coverage {window.describe()['median_coverage']:.2f}, "
            f"modelled spread {window.modelled_relative_spread:.3f} per leg"
        )
        print(
            "  These sessions were reconstructed, not recorded. The book for them "
            "is gone: the spread is modelled rather than measured, the liquidity "
            "and open-interest features are absent, and entry prices are prints."
        )
        snapshots = window.snapshots
        closes = window.settlements
        build_features = window.feature_builder()
        names_for = training.reconstructed_feature_names_for
        provenance = "reconstructed"
    else:
        recorded = archive.sessions(chains)
        if not recorded:
            print(
                "no recorded chains to replay. The agent archives each 10:00 snapshot "
                "as it runs; there are none yet. Pass --reconstructed to replay "
                "sessions rebuilt from the option tape instead."
            )
            return 1
        snapshots = list(archive.read_all(chains))
        closes = settlements(gateway, symbol, recorded, zone)
        build_features = None

    samples = training.build_samples(
        snapshots, closes, scenarios, config, rank,
        build_features=build_features, label_top_k=arguments.label_top_k,
    )
    if not samples:
        print("no session could be labelled; nothing to replay")
        return 1

    probabilities = out_of_sample_probabilities(samples, config, names_for)
    report = backtest.run(samples, probabilities, config.float_("risk.es_confidence"))

    sessions = report.sessions
    print(f"\nreplayed {sessions} {provenance} session(s)\n")
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

    if arguments.json is not None:
        destination = (
            pathlib.Path(arguments.json) if arguments.json
            else config.path_("paths.backtest_report")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = report.as_dict()
        payload["note"] = (
            "Replayed on chains recorded by the agent at 10:00 ET. Sharpe figures over "
            f"{sessions} sessions are arithmetic, not evidence."
        )
        destination.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote the full report to {destination}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
