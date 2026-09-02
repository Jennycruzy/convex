"""Measure what execution actually costs on SPY 0DTE, family by family.

The research is SPX: five-dollar strikes on a 6,800 index. This trades SPY:
one-dollar strikes on a 650 ETF. A spread copied across in points would be a
different trade, so every cost figure in the configuration starts life marked
as a hypothesis and is replaced by what this script measures.

For each of the five families it builds the candidates the agent would build,
prices them from live quotes, and reports the distribution of what it costs to
cross: per leg, per structure, and as a share of the structure's own worst case.
The last of those is the number that matters, because a cost of forty dollars
means one thing on a structure risking five hundred and another on one risking
eighty.

By default it measures and prints. With --write it also puts the one figure it
can honestly measure into config/convex.yaml and clears that key from the
blocking provenance list, which is the step that would otherwise be a hand edit
minutes before an entry.

Only `liquidity.max_relative_spread` is written, because it is the only one of
the checked inputs a chain snapshot can settle. Slippage is a fill against the
mid it was sent at, both fees arrive on the account's activity record, and the
pin band needs a real close watched. Those stay conservative bounds until a
trade exists to measure them from.

Run it with:  .venv/bin/python -m scripts.calibrate_costs
              .venv/bin/python -m scripts.calibrate_costs --write
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from convex.config import load
from convex.costs import CostModel
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConfigError, ConvexError, UndefinedRiskError
from convex.measured import refresh_measurement, write_atomically
from convex.ledger import Action, Ledger, Record, new_cycle_id
from convex.payoff import risk_profile
from convex.structures import build_candidates

REPO_ROOT = Path(__file__).resolve().parent.parent

# The one checked input a chain snapshot can settle on its own.
MEASURED_KEY = "liquidity.max_relative_spread"

# Below this many quoted legs a median is three contracts and a rounding error,
# so the run reports and refuses rather than writing a threshold off it.
MINIMUM_LEGS = 40


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="put the measured liquidity threshold into config/convex.yaml and "
        "clear it from provenance.hypothesis",
    )
    arguments = parser.parse_args()

    config = load()
    symbol = config.str_("underlying.symbol")
    gateway = AlpacaGateway(config)
    ledger = Ledger(config.path_("paths.ledger"))
    model = CostModel.from_config(config)

    now, market_open = gateway.clock()
    expiries = gateway.expirations(symbol, now.date())
    expiry = expiries[0]
    spot, _ = gateway.spot(symbol)

    low = spot * config.float_("candidates.moneyness_low")
    high = spot * config.float_("candidates.moneyness_high")
    wing = spot * config.float_("candidates.max_wing_width_pct")
    chain = gateway.chain(symbol, expiry, low - wing, high + wing)

    leg_half_spreads = [row.quote.half_spread for row in chain]
    leg_relative = [row.quote.relative_spread for row in chain]

    # A leg quoted on one side only is not an expensive leg, it is a leg nobody
    # is making a market in. With bid at zero the mid is half the ask and the
    # relative spread computes to exactly 2.0, which is where the 200% p90 in
    # every one of these notes comes from. Leaving those in the population that
    # sets the threshold drags the median up and makes the liquidity check more
    # permissive, which is the same defect as taking the p90 one layer down:
    # the number ends up describing the book instead of constraining it. They
    # stay in the printed distribution, because what the whole band looks like
    # is worth seeing, and they are excluded from the threshold.
    tradeable = tradeable_legs(chain)
    leg_relative_tradeable = [row.quote.relative_spread for row in tradeable]

    print(f"{symbol} at {spot:,.2f}, expiry {expiry}, {len(chain)} contracts\n")
    print("per leg, across the whole snapshot")
    print(f"  half-spread      median {statistics.median(leg_half_spreads):.3f} "
          f"  p90 {_quantile(leg_half_spreads, 0.9):.3f}")
    print(f"  relative spread  median {statistics.median(leg_relative):.1%} "
          f"  p90 {_quantile(leg_relative, 0.9):.1%}")
    one_sided = len(chain) - len(tradeable)
    if leg_relative_tradeable:
        print(f"  two sided only   median {statistics.median(leg_relative_tradeable):.1%} "
              f"  p90 {_quantile(leg_relative_tradeable, 0.9):.1%} "
              f"  ({one_sided} of {len(chain)} legs quoted one sided)")

    rows: list[dict] = []
    print("\nper structure, at one contract")
    for family, candidates in build_candidates(chain, config, spot).items():
        totals, shares, skipped = [], [], 0
        for candidate in candidates:
            breakdown = model.breakdown(candidate.legs, 1)
            try:
                profile = risk_profile(
                    candidate.legs, model.risk_debit(candidate.legs, 1)
                )
            except UndefinedRiskError:
                # A structure whose worst case is not computable is not costed;
                # it would never reach an order either.
                skipped += 1
                continue
            totals.append(breakdown.total)
            shares.append(breakdown.total / profile.max_loss)

        if not totals:
            print(f"  {family:<16} no priceable candidate")
            continue

        entry = {
            "family": str(family),
            "candidates": len(totals),
            "skipped": skipped,
            "median_cost": round(statistics.median(totals), 2),
            "p90_cost": round(_quantile(totals, 0.9), 2),
            "median_cost_share_of_max_loss": round(statistics.median(shares), 4),
        }
        rows.append(entry)
        print(
            f"  {family:<16} {len(totals):>4} candidates   "
            f"cost median {entry['median_cost']:>7.2f}   p90 {entry['p90_cost']:>7.2f}   "
            f"median cost is {entry['median_cost_share_of_max_loss']:.1%} of the worst case"
        )

    measured = {
        # Whether anyone was quoting when this was taken. A spread measured off
        # a shut book is a real number about nothing, and the record has to say
        # so: this note and the ledger entry are published on the dashboard.
        "market_open": market_open,
        "spot": round(spot, 2),
        "expiry": expiry.isoformat(),
        "contracts": len(chain),
        "leg_half_spread_median": round(statistics.median(leg_half_spreads), 4),
        "leg_half_spread_p90": round(_quantile(leg_half_spreads, 0.9), 4),
        "leg_relative_spread_median": round(statistics.median(leg_relative), 4),
        "leg_relative_spread_p90": round(_quantile(leg_relative, 0.9), 4),
        "two_sided_legs": len(tradeable),
        "one_sided_legs": len(chain) - len(tradeable),
        "leg_relative_spread_median_two_sided": (
            round(statistics.median(leg_relative_tradeable), 4) if leg_relative_tradeable else None
        ),
        "families": rows,
    }

    ledger.append(
        Record(
            action=Action.CALIBRATION,
            cycle_id=new_cycle_id(),
            rationale=(
                f"Measured SPY 0DTE execution cost on the {expiry} expiry with the "
                f"underlying at {spot:,.2f}: the median leg costs "
                f"{measured['leg_half_spread_median']:.3f} per share to cross."
                + ("" if market_open else " Taken with the market closed, so these "
                   "are the spreads of a book nobody is quoting and no threshold "
                   "was written from them.")
            ),
            outcome=measured,
        )
    )
    _write_note(measured)
    print(f"\nwritten to the ledger and to docs/calibration-{datetime.now(timezone.utc):%Y-%m-%d}.md")

    # The median, not the p90. A threshold set at the ninetieth percentile
    # admits ninety percent of the legs it measured, which describes the book
    # instead of constraining it: on 31 August the p90 was 200% and the value
    # it proposed rejected nothing at all. tests/test_gates.py caught that,
    # with 1551 of 1551 candidates surviving a check that is supposed to bind.
    #
    # This threshold is not the profitability test and was never meant to be.
    # Cost is priced into every candidate before ranking, so a structure that
    # cannot pay for its own execution is already last. What this rejects is a
    # leg whose book is not a market: wider than the typical leg in the band.
    threshold = _quantile(leg_relative_tradeable, 0.5) if leg_relative_tradeable else None
    if threshold is None:
        print("\nrefusing to write: not one leg in the band is quoted on both sides.")
        return 2
    if not arguments.write:
        print(f"\nmeasured  {MEASURED_KEY}: {threshold:.4f}")
        print("re-run with --write to put it in the configuration")
        return 0

    # The count that gates the write is the tradeable one, because that is the
    # population the median is taken over now.
    return _apply(config, ledger, threshold, len(tradeable), market_open, measured)


def _apply(config, ledger, threshold: float, legs: int, market_open: bool, measured: dict) -> int:
    """Write the threshold, or refuse and say why, having changed nothing.

    Both refusals are about the same thing. A relative spread read off a shut
    market is the width of a book nobody is maintaining, and one read off a
    handful of legs is not a distribution. Either would be baked into a live
    threshold and then trusted as a measurement.
    """
    if not market_open:
        print(
            "\nrefusing to write: the market is closed, so these are the spreads of "
            "a book nobody is quoting. Run this during a session."
        )
        return 2
    if legs < MINIMUM_LEGS:
        print(
            f"\nrefusing to write: {legs} quoted legs is too few for a median to "
            f"mean anything (wanted {MINIMUM_LEGS})."
        )
        return 2

    before = config.float_(MEASURED_KEY)
    path = config.path
    try:
        updated = refresh_measurement(
            path.read_text(),
            MEASURED_KEY,
            threshold,
            datetime.now(timezone.utc).date(),
            "scripts/calibrate_costs.py",
        )
    except ConfigError as error:
        print(f"\nrefusing to write: {error}")
        return 2

    write_atomically(path, updated)
    ledger.append(
        Record(
            action=Action.CALIBRATION,
            cycle_id=new_cycle_id(),
            rationale=(
                f"Measured {MEASURED_KEY} at {threshold:.4f} from {legs} quoted legs "
                f"and replaced the standing value of {before:.4f}. The key is cleared "
                "from provenance.hypothesis, so the calibration check no longer "
                "stands the session down for it."
            ),
            outcome={
                "key": MEASURED_KEY,
                "before": before,
                "after": round(threshold, 6),
                "legs": legs,
                "market_open": market_open,
                "measurement": measured,
            },
        )
    )
    print(f"\n{MEASURED_KEY}: {before:.4f} -> {threshold:.4f}, written to {path}")
    print("cleared from provenance.hypothesis; the session will price on it")
    return 0


def tradeable_legs(chain):
    """The legs of a chain that someone is quoting on both sides.

    A leg with no bid cannot be sold and a leg with no ask cannot be bought, so
    neither belongs in a distribution used to decide what a normal leg costs to
    cross. With bid at zero the mid is half the ask and the relative spread is
    exactly 2.0 by construction, which is a fact about the absence of a market
    rather than a measurement of one.
    """
    return [row for row in chain if row.quote.bid > 0.0 and row.quote.ask > 0.0]


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def _write_note(measured: dict) -> None:
    stamp = datetime.now(timezone.utc)
    path = REPO_ROOT / "docs" / f"calibration-{stamp:%Y-%m-%d}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# SPY 0DTE execution cost, measured {stamp:%Y-%m-%d %H:%M UTC}",
        "",
        f"Underlying at {measured['spot']:,.2f}, expiry {measured['expiry']}, "
        f"{measured['contracts']} contracts in the band.",
        "",
        (
            "Taken during a session."
            if measured.get("market_open")
            else "**Taken with the market closed.** These are the spreads of a book "
            "nobody is quoting. Nothing was written into the configuration from "
            "them and nothing should be."
        ),
        "",
        "## Per leg",
        "",
        "| | median | p90 |",
        "|---|---|---|",
        f"| half-spread | {measured['leg_half_spread_median']:.3f} | "
        f"{measured['leg_half_spread_p90']:.3f} |",
        f"| relative spread | {measured['leg_relative_spread_median']:.1%} | "
        f"{measured['leg_relative_spread_p90']:.1%} |",
        "",
        "## Per structure, one contract",
        "",
        "| family | candidates | median cost | p90 cost | median cost as a share of the worst case |",
        "|---|---|---|---|---|",
    ]
    for row in measured["families"]:
        lines.append(
            f"| {row['family']} | {row['candidates']} | {row['median_cost']:.2f} | "
            f"{row['p90_cost']:.2f} | {row['median_cost_share_of_max_loss']:.1%} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
