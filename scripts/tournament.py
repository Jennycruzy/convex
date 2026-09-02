"""Run isolated CONVEX tournament profiles, dry-run by default.

By default every read is live paper-market data and every broker write is
blocked by ``DryRunGateway``. Explicit submissions may send one bounded,
defined-risk order to the configured Alpaca *paper* account. ``execution_bwb``
may retry a canceled, zero-fill order twice from fresh quotes and fresh gates.

Run: .venv/bin/python -m scripts.tournament
     .venv/bin/python -m scripts.tournament --profile skew_bwb
     .venv/bin/python -m scripts.tournament --submit --profile execution_bwb
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict

from convex.agent import Agent
from convex.classifier import load_models
from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError
from convex.ledger import Action, Ledger, Record, new_cycle_id
from convex.scenarios import build as build_scenarios
from convex.tournament import (
    fill_ladder,
    intraday_trend,
    profile_config,
    profiles,
    submission_config,
    trend_candidate_filter,
)
from convex.variance import load_history
from scripts.run_cycle import SUBMISSION_CUTOFF, DryRunGateway, family_results


def _record_profile_start(
    ledger: Ledger, name: str, description: str, context: dict, *, submit: bool
) -> None:
    mode = (
        "is authorized to submit one defined-risk paper structure through the canonical ledger"
        if submit
        else "is an isolated dry run whose gateway refuses every order submission"
    )
    ledger.append(Record(
        action=Action.CALIBRATION,
        cycle_id=new_cycle_id(),
        rationale=f"Tournament profile {name!r} {mode}. {description}",
        extra=context,
    ))


def _record_flat_trend(ledger: Ledger, context: dict, detail: str) -> None:
    ledger.append(Record(
        action=Action.STAND_DOWN,
        cycle_id=new_cycle_id(),
        structure="debit_vertical",
        rationale=(
            "Trend vertical stood down: no directional vertical was priced because " + detail + "."
        ),
        reject_reason="trend_not_confirmed",
        extra=context,
    ))


def _record_ladder(ledger: Ledger, cycle_id: str, profile, config, context: dict) -> None:
    if not profile.fill_ladder_ticks:
        return
    records = list(ledger.read())
    planned = [
        row for row in records
        if row.get("cycle_id") == cycle_id and row.get("action") == Action.DRY_RUN.value
    ]
    for row in planned:
        initial = float(row["net_price"])
        ladder = fill_ladder(initial, profile.fill_ladder_ticks, config.float_("costs.tick_size"))
        ledger.append(Record(
            action=Action.CALIBRATION,
            cycle_id=cycle_id,
            structure=str(row.get("structure") or ""),
            rationale=(
                "Execution observation only: no order was sent. If execution is later "
                "authorized, every ladder rung must be re-priced and must still clear "
                "the net-edge and all risk gates."
            ),
            extra={**context, "execution_observation": {
                "initial_limit": initial,
                "ladder_limits": ladder,
                "ticks": list(profile.fill_ladder_ticks),
                "submission_enabled": False,
            }},
        ))


def run_profile(live: AlpacaGateway, base_config, scenarios, history, profile, *, submit: bool) -> None:
    config = submission_config(base_config, profile) if submit else profile_config(base_config, profile)
    ledger = Ledger(config.path_("paths.ledger"))
    mode = "paper_submission" if submit else "dry_run_only"
    context = {"tournament": {"profile": profile.name, "mode": mode}}
    _record_profile_start(ledger, profile.name, profile.description, context, submit=submit)

    now, _ = live.clock()
    candidate_filter = None
    if profile.name == "trend_vertical":
        signal = intraday_trend(live, config, now)
        context["tournament"]["trend"] = asdict(signal)
        if not signal.tradeable:
            _record_flat_trend(ledger, context, signal.detail)
            print(f"{profile.name}: {signal.detail}; no candidate was priced")
            return
        candidate_filter = trend_candidate_filter(signal)
        print(f"{profile.name}: {signal.detail}")

    models, _ = load_models(config.path_("paths.models"), config)
    agent = Agent(
        gateway=live if submit else DryRunGateway(live),
        config=config,
        ledger=ledger,
        scenarios=scenarios,
        models=models,
        submission_cutoff=SUBMISSION_CUTOFF,
        dry_run=not submit,
        candidate_filter=candidate_filter,
        receipt_context=context,
        reprice_ticks=tuple(profile.fill_ladder_ticks[1:]) if submit and profile.name == "execution_bwb" else (),
    )
    result = agent.run_cycle(
        prior_returns=scenarios.log_returns.tolist(),
        variance_history=history.as_list(),
        family_pnl=family_results(ledger),
    )
    if not submit:
        _record_ladder(ledger, result.cycle_id, profile, config, context)
    print(f"{profile.name}: {result.reason} ({config.path_('paths.ledger')})")
    for order in result.orders:
        prefix = "submitted" if submit else "dry-run only"
        print(f"  {prefix}: {order['family']} {order['order_id']}")


def main() -> int:
    base_config = load()
    available = profiles(base_config)
    names = [profile.name for profile in available]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["all", *names], default="all")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="submit one bounded BWB profile to the configured Alpaca paper account",
    )
    arguments = parser.parse_args()
    if arguments.submit and arguments.profile not in {"skew_bwb", "execution_bwb"}:
        parser.error("--submit requires skew_bwb or execution_bwb; trend_vertical is observation-only")

    chosen = available if arguments.profile == "all" else tuple(
        profile for profile in available if profile.name == arguments.profile
    )
    live = AlpacaGateway(base_config)
    # In dry-run mode every Agent receives DryRunGateway, whose
    # submit_structure method unconditionally raises instead of writing. The
    # explicit BWB submission paths retain the canonical ledger, all
    # session/candidate gates, and a one-structure concurrency cap.
    scenarios = build_scenarios(live, base_config)
    now, _ = live.clock()
    history = load_history(
        base_config.path_("paths.variance_history"),
        base_config.str_("underlying.symbol"),
        now.date(),
        base_config.int_("classifier.variance_history_max_age_days"),
        base_config.int_("classifier.variance_history_min_readings"),
    )
    print(f"scenario set: {scenarios.describe()}")
    if arguments.submit:
        print("TOURNAMENT MODE: ONE GUARDED PAPER SUBMISSION")
    else:
        print("TOURNAMENT MODE: DRY RUN ONLY — no broker order can be sent")
    for profile in chosen:
        run_profile(live, base_config, scenarios, history, profile, submit=arguments.submit)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
