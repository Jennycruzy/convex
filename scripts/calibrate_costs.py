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

Run it with:  .venv/bin/python -m scripts.calibrate_costs
"""

from __future__ import annotations

import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from convex.config import load
from convex.costs import CostModel
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError, UndefinedRiskError
from convex.ledger import Action, Ledger, Record, new_cycle_id
from convex.payoff import risk_profile
from convex.structures import build_candidates

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    config = load()
    symbol = config.str_("underlying.symbol")
    gateway = AlpacaGateway(config)
    ledger = Ledger(config.path_("paths.ledger"))
    model = CostModel.from_config(config)

    now, _ = gateway.clock()
    expiries = gateway.expirations(symbol, now.date())
    expiry = expiries[0]
    spot, _ = gateway.spot(symbol)

    low = spot * config.float_("candidates.moneyness_low")
    high = spot * config.float_("candidates.moneyness_high")
    wing = spot * config.float_("candidates.max_wing_width_pct")
    chain = gateway.chain(symbol, expiry, low - wing, high + wing)

    leg_half_spreads = [row.quote.half_spread for row in chain]
    leg_relative = [row.quote.relative_spread for row in chain]

    print(f"{symbol} at {spot:,.2f}, expiry {expiry}, {len(chain)} contracts\n")
    print("per leg, across the whole snapshot")
    print(f"  half-spread      median {statistics.median(leg_half_spreads):.3f} "
          f"  p90 {_quantile(leg_half_spreads, 0.9):.3f}")
    print(f"  relative spread  median {statistics.median(leg_relative):.1%} "
          f"  p90 {_quantile(leg_relative, 0.9):.1%}")

    rows: list[dict] = []
    print("\nper structure, at one contract")
    for family, candidates in build_candidates(chain, config, spot).items():
        totals, shares, skipped = [], [], 0
        for candidate in candidates:
            breakdown = model.breakdown(candidate.legs, 1)
            try:
                profile = risk_profile(
                    candidate.legs, model.executable_debit(candidate.legs, 1)
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
        "spot": round(spot, 2),
        "expiry": expiry.isoformat(),
        "contracts": len(chain),
        "leg_half_spread_median": round(statistics.median(leg_half_spreads), 4),
        "leg_half_spread_p90": round(_quantile(leg_half_spreads, 0.9), 4),
        "leg_relative_spread_median": round(statistics.median(leg_relative), 4),
        "leg_relative_spread_p90": round(_quantile(leg_relative, 0.9), 4),
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
            ),
            outcome=measured,
        )
    )
    _write_note(measured)
    print(f"\nwritten to the ledger and to docs/calibration-{datetime.now(timezone.utc):%Y-%m-%d}.md")
    print("suggested configuration, to be applied by hand after reading it:")
    print(f"  liquidity.max_relative_spread: {_quantile(leg_relative, 0.9):.2f}")
    return 0


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
