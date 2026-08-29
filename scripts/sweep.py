"""Replay the rebuilt sessions at a range of spreads and find where net edge dies.

The strategy's whole question is not whether it makes money gross — it does —
but whether it survives what it costs to trade. That answer is a function of
one number, the spread paid per leg, so this measures the answer across a range
of that number instead of asserting it at one.

Every point here is a full replay, not an interpolation. The curve the
dashboard draws is these points and nothing between them.

Run it with:  .venv/bin/python -m scripts.sweep
"""

from __future__ import annotations

import argparse
import json
import sys

from convex import backtest, reconstruct, training
from convex.agent import rank
from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError
from convex.scenarios import build as build_scenarios
from scripts.backtest import out_of_sample_probabilities

SPREADS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=400)
    arguments = parser.parse_args()

    config = load()
    gateway = AlpacaGateway(config)
    scenarios = build_scenarios(gateway, config)
    confidence = config.float_("risk.es_confidence")
    names_for = training.reconstructed_feature_names_for

    points = []
    for spread in SPREADS:
        window = reconstruct.rebuild_window(gateway, config, arguments.days, spread)
        samples = training.build_samples(
            window.snapshots,
            window.settlements,
            scenarios,
            config,
            rank,
            build_features=window.feature_builder(),
        )
        probabilities = out_of_sample_probabilities(samples, config, names_for)
        report = backtest.run(samples, probabilities, confidence)
        classified = report.basket["classified"]
        everyday = report.basket["every session"]
        points.append(
            {
                "relative_spread": spread,
                "sessions": report.sessions,
                "classified": {
                    "trades": classified.trades,
                    "gross_sharpe": classified.gross_sharpe,
                    "net_sharpe": classified.net_sharpe,
                    "gross": classified.gross_total,
                    "net": classified.net_total,
                    "cost": classified.cost_total,
                },
                "every_session": {
                    "trades": everyday.trades,
                    "gross_sharpe": everyday.gross_sharpe,
                    "net_sharpe": everyday.net_sharpe,
                    "gross": everyday.gross_total,
                    "net": everyday.net_total,
                    "cost": everyday.cost_total,
                },
            }
        )
        print(
            f"  spread {spread:.3f}  classified gross "
            f"{classified.gross_sharpe if classified.gross_sharpe is not None else float('nan'):.2f} "
            f"net {classified.net_sharpe if classified.net_sharpe is not None else float('nan'):.2f}",
            flush=True,
        )

    path = config.path_("paths.backtest_report").parent / "sensitivity.json"
    path.write_text(
        json.dumps(
            {
                "reconstructed": True,
                "days": arguments.days,
                "note": (
                    "Each point is a full replay of sessions rebuilt from the option "
                    "tape at that modelled spread. The spread is modelled, not "
                    "measured: the book for those sessions is gone."
                ),
                "points": points,
            },
            indent=2,
        )
    )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
