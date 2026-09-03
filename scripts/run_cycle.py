"""Run one decision cycle against the live paper account.

This is what runs at 10:00 ET, the entry time the research specifies and holds
to the close. It builds the scenario set from SPY's own recent history, reads
the prior sessions and each family's own past results out of the ledger, and
hands all of it to the agent.

Run it with:  .venv/bin/python -m scripts.run_cycle
Add --dry-run to do everything except send an order.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from convex.agent import Agent
from convex.classifier import load_models
from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError
from convex.ledger import Action, Ledger, Record, new_cycle_id
from convex.scenarios import build as build_scenarios
from convex.scenarios import save as save_scenarios
from convex.variance import load_history

# Submissions close at 15:00 UTC on 4 September, which is before the opening
# bell in New York, so no position is opened that morning.
SUBMISSION_CUTOFF = datetime(2026, 9, 4, 11, 0, tzinfo=ZoneInfo("America/New_York"))


class DryRunGateway:
    """Everything the live gateway does, except sending an order.

    This is not a mock of Alpaca: every read goes to the real API. Only the
    write is withheld, so a dry run exercises the same chain, the same account
    and the same checks as a live cycle.
    """

    def __init__(self, gateway: AlpacaGateway) -> None:
        self._gateway = gateway

    def __getattr__(self, name):
        return getattr(self._gateway, name)

    def submit_structure(self, legs, contracts, limit_price, client_order_id):
        raise ConvexError(
            f"dry run: would have sent {contracts} lots of a {len(legs)}-leg structure "
            f"at a limit of {limit_price:.2f} as {client_order_id}"
        )


def family_results(ledger: Ledger) -> dict[str, list[float]]:
    """Each family's realised results so far, oldest first."""
    history: dict[str, list[float]] = defaultdict(list)
    records = list(ledger.read())
    invalidated = {
        int(sequence)
        for record in records
        if record.get("action") == Action.CORRECTION.value
        for sequence in record.get("invalidates", [])
        if str(sequence).isdigit()
    }
    for record in records:
        if record.get("seq") in invalidated:
            continue
        if record.get("action") != Action.POSITION_CLOSED.value:
            continue
        outcome = record.get("outcome") or {}
        if "realised_pnl" in outcome and record.get("structure"):
            history[record["structure"]].append(float(outcome["realised_pnl"]))
    return dict(history)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="price and check, but do not send")
    arguments = parser.parse_args()

    config = load()
    if config.str_("strategy.active_profile") == "gap_continuation_vertical":
        # Keep the installed 10:00 ET service, but never let that generic
        # entrypoint enumerate debit verticals. The isolated runner enables
        # the sole family only in memory after its deterministic signal fires.
        from scripts.run_gap_continuation import main as run_gap_continuation

        return run_gap_continuation()
    live = AlpacaGateway(config)
    ledger = Ledger(config.path_("paths.ledger"))

    scenarios = build_scenarios(live, config)
    save_scenarios(scenarios, config.path_("paths.scenario_archive"))
    print(f"scenario set: {scenarios.describe()}")

    now, _ = live.clock()
    sessions = live.sessions(now.date(), now.date())
    if not sessions:
        # Law 4: a refusal is a decision and leaves the same receipt as a trade.
        # The cycle returns here before the agent runs, so the record it would
        # otherwise have written has to be written here instead.
        ledger.append(
            Record(
                action=Action.STAND_DOWN,
                cycle_id=new_cycle_id(),
                rationale=(
                    f"No trade on {now.date()}: Alpaca's calendar lists no session. "
                    "Nothing was priced and nothing was sent."
                ),
                reject_reason="market_calendar",
            )
        )
        print(f"no session on {now.date()}; nothing to do")
        return 0

    models, metadata = load_models(config.path_("paths.models"), config)
    if models:
        fitted = ", ".join(sorted(str(name) for name in models))
        print(f"classifiers loaded ({metadata.get('fitted_at', 'unknown date')}): {fitted}")
    else:
        print(
            "no fitted classifiers on disk; every family falls back to the "
            "documented volatility-regime rule, and each record says so"
        )

    agent = Agent(
        # Both, on purpose. The flag makes the agent record a rehearsal as one
        # instead of writing an order into the evidence, and the wrapper stays
        # underneath it as the thing that physically cannot send.
        gateway=DryRunGateway(live) if arguments.dry_run else live,
        config=config,
        ledger=ledger,
        scenarios=scenarios,
        models=models,
        submission_cutoff=SUBMISSION_CUTOFF,
        dry_run=arguments.dry_run,
    )

    # Implied against implied. The rule used to be handed realised variance out
    # of the scenario set, because a recorded implied series did not exist when
    # it was written. It does now, rebuilt off the tape, and the substitute had
    # a direction: 489 of 550 rebuilt sessions read high_variance against it,
    # one read low_variance, and the two families favoured in that regime could
    # never be selected. See convex/variance.py for what this refuses to read.
    #
    # Law 4 reaches this too. Every refusal above writes a receipt before it
    # returns, and a history this refuses to read is a refusal like any other:
    # without a record it would be a day the agent did not trade and left
    # nothing saying why, which is the one shape of silence this ledger exists
    # to make impossible.
    try:
        history = load_history(
            config.path_("paths.variance_history"),
            config.str_("underlying.symbol"),
            now.date(),
            config.int_("classifier.variance_history_max_age_days"),
            config.int_("classifier.variance_history_min_readings"),
        )
    except ConvexError as error:
        ledger.append(
            Record(
                action=Action.RISK_HALT,
                cycle_id=new_cycle_id(),
                rationale=(
                    f"No entry on {now.date()}: the regime rule has no history it can "
                    f"trust, so no direction was called and nothing was priced. {error}"
                ),
                reject_reason="variance_history",
            )
        )
        raise
    print(
        f"regime yardstick: {len(history)} prior implied readings, "
        f"{history.first_session.isoformat()} to {history.last_session.isoformat()}"
    )

    result = agent.run_cycle(
        prior_returns=scenarios.log_returns.tolist(),
        variance_history=history.as_list(),
        family_pnl=family_results(ledger),
    )

    print(f"\ncycle {result.cycle_id}: {result.reason}")
    for rejection in result.rejections:
        print(f"  refused {rejection['family']:<16} {rejection['reason']}: {rejection['detail']}")
    for order in result.orders:
        verb = "would open" if arguments.dry_run else "opened  "
        print(f"  {verb} {order['family']:<16} order {order['order_id']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
