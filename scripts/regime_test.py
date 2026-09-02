"""Does changing the regime yardstick make money, or only change the histogram?

The volatility regime rule compares today's implied variance against a history
of realised variance, because recorded implied variance did not exist when the
rule was written. Across 550 rebuilt sessions the median implied reading sits
at the 75.7th percentile of the realised distribution, so the comparison is
biased: 489 of 550 sessions read high_variance and one reads low_variance,
which leaves call_bwb and debit_vertical unselectable.

scripts/variance_history.py rebuilt the implied series that removes the bias.
That the distribution then lands near the 40/20/40 the band is built for is not
by itself a reason to ship it. A rule can be better shaped and still lose more
money. This replays the same sessions three ways and reports what each earns
net of cost:

  realised   the yardstick running today
  implied    the same rule against prior implied readings
  classifier the fitted model, for reference, which does not ship

Nothing here writes to the configuration or the ledger.

Run it with:  .venv/bin/python -m scripts.regime_test --days 800 --relative-spread 0.041
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from convex import backtest, reconstruct, training
from convex.agent import rank
from convex.classifier import RegimeRule
from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError, DataError
from convex.scenarios import build as build_scenarios
from convex.structures.base import Family
from scripts.backtest import out_of_sample_probabilities, show


def regime_probabilities(
    sessions: list[date],
    implied: dict[date, float],
    history_for,
    families: list[Family],
    rule: RegimeRule,
) -> tuple[dict, dict]:
    """Probability per family per session, and the regime that produced it."""
    probabilities: dict[Family, dict[date, float]] = {family: {} for family in families}
    regimes: dict[date, str] = {}
    for day in sessions:
        reading = implied.get(day)
        if reading is None:
            continue
        history = history_for(day)
        if len(history) < 5:
            continue
        regime = rule.regime(reading, history)
        regimes[day] = regime
        for family in families:
            probabilities[family][day] = rule.probability(family, regime)
    return probabilities, regimes


def arm(report, label: str) -> str:
    basket = report.basket.get("classified")
    if basket is None:
        return f"{label:12} no sessions taken"
    return (
        f"{label:12}{basket.trades:>7}{show(basket.gross_sharpe)}{show(basket.net_sharpe)}"
        f"{basket.net_total:>13,.2f}{basket.cost_total:>13,.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=800)
    parser.add_argument("--relative-spread", type=float, default=0.041)
    parser.add_argument("--history", default="data/variance/implied.json")
    parser.add_argument("--label-top-k", type=int, default=1)
    arguments = parser.parse_args()

    config = load()
    gateway = AlpacaGateway(config)
    scenarios = build_scenarios(gateway, config)

    path = Path(arguments.history)
    if not path.exists():
        raise DataError(
            f"{path} does not exist. Build it with scripts/variance_history.py before "
            "asking what the implied yardstick would have earned."
        )
    series = json.loads(path.read_text())["series"]
    implied = {date.fromisoformat(row["session"]): row["iv_total"] for row in series}
    ordered = sorted(implied)
    print(f"{len(implied)} implied variance readings from {ordered[0]} to {ordered[-1]}")

    window = reconstruct.rebuild_window(
        gateway, config, arguments.days, arguments.relative_spread
    )
    if not window.snapshots:
        raise DataError("no session could be rebuilt; nothing to replay")
    print(
        f"rebuilt {len(window.snapshots)} sessions, modelled spread "
        f"{window.modelled_relative_spread:.3f} per leg\n"
    )

    samples = training.build_samples(
        window.snapshots, window.settlements, scenarios, config, rank,
        build_features=window.feature_builder(), label_top_k=arguments.label_top_k,
    )
    if not samples:
        raise DataError("no session could be labelled; nothing to replay")

    families = [Family(str(name)) for name in config.list_("structures.enabled")]
    rule = RegimeRule()
    days = sorted({sample.session_date for sample in samples})

    # The yardstick running today: one realised distribution, the same for every
    # session, exactly as the live cycle passes it.
    realised = scenarios.annualised_variance().tolist()
    by_realised, regimes_realised = regime_probabilities(
        days, implied, lambda _day: realised, families, rule
    )

    # The proposed yardstick: only readings from strictly earlier sessions, so
    # no session is judged against its own future.
    def prior_implied(day: date) -> list[float]:
        return [implied[d] for d in ordered if d < day]

    by_implied, regimes_implied = regime_probabilities(
        days, implied, prior_implied, families, rule
    )

    for label, regimes in (("realised", regimes_realised), ("implied", regimes_implied)):
        counts: dict[str, int] = {}
        for value in regimes.values():
            counts[value] = counts.get(value, 0) + 1
        total = sum(counts.values()) or 1
        shares = "  ".join(
            f"{name} {counts.get(name, 0):>4} {counts.get(name, 0) / total:5.1%}"
            for name in ("low_variance", "middle", "high_variance")
        )
        print(f"regime under {label:9} {shares}")
    print()

    print(f"{'yardstick':12}{'trades':>7}{'gross':>8}{'net':>8}{'net total':>13}{'cost':>13}")
    print("-" * 61)
    print(arm(backtest.run(samples, by_realised, config.float_("risk.es_confidence")), "realised"))
    print(arm(backtest.run(samples, by_implied, config.float_("risk.es_confidence")), "implied"))
    reference = backtest.run(
        samples,
        out_of_sample_probabilities(samples, config, training.reconstructed_feature_names_for),
        config.float_("risk.es_confidence"),
    )
    print(arm(reference, "classifier"))
    print("\nclassifier is shown for reference and does not ship; neither bar was cleared.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
