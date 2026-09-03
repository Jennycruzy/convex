"""Run the one-lot, deterministic 10:00 ET gap-continuation profile.

The production configuration remains entry-disabled.  This runner enables only
``debit_vertical`` in memory after the signal is observed, then lets the normal
Agent apply its full session and candidate-gate stack.  It never retries an
unfilled order and it cannot place more than one contract.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, time, timedelta
from time import sleep
from zoneinfo import ZoneInfo

from convex.agent import Agent
from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError, DataError
from convex.gap_continuation import signal
from convex.ledger import Action, Ledger, Record, new_cycle_id
from convex.scenarios import build as build_scenarios
from convex.scenarios import save as save_scenarios

SUBMISSION_CUTOFF = datetime(2026, 9, 4, 11, 0, tzinfo=ZoneInfo("America/New_York"))
PROFILE_PROBABILITY = 0.75
PROFILE_SOURCE = "gap-continuation profile: 9/12 positive reconstructed held-out verticals"


class DryRunGateway:
    """Forward every read to Alpaca while physically withholding order writes."""

    def __init__(self, gateway):
        self._gateway = gateway

    def __getattr__(self, name):
        return getattr(self._gateway, name)

    def submit_structure(self, legs, contracts, limit_price, client_order_id):
        raise ConvexError(
            f"dry run: would have sent {contracts} lots of a {len(legs)}-leg structure "
            f"at a limit of {limit_price:.2f} as {client_order_id}"
        )


def family_results(ledger: Ledger) -> dict[str, list[float]]:
    """Only reconciled, non-invalidated closes are usable as family history."""
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
        if record.get("seq") in invalidated or record.get("action") != Action.POSITION_CLOSED.value:
            continue
        outcome = record.get("outcome") or {}
        if record.get("structure") and "realised_pnl" in outcome:
            history[str(record["structure"])] .append(float(outcome["realised_pnl"]))
    return dict(history)


def observed_signal(gateway: AlpacaGateway, now: datetime, minimum_gap: float):
    """Read the completed 10:00 bar and the preceding session close."""
    zone = ZoneInfo("America/New_York")
    if now.tzinfo is None:
        raise DataError("exchange clock must carry a timezone")
    local_now = now.astimezone(zone)
    if not time(10, 1) <= local_now.time() <= time(10, 3):
        raise DataError("gap profile only permits execution from 10:01 to 10:03 ET")
    sessions = gateway.sessions(local_now.date() - timedelta(days=10), local_now.date())
    dates = [session.session_date for session in sessions]
    previous = [day for day in dates if day < local_now.date()]
    if local_now.date() not in dates or not previous:
        raise DataError("no current and prior Alpaca trading session for gap signal")
    prior_day = previous[-1]
    start = datetime.combine(prior_day, time(9, 30), tzinfo=zone)
    bars = gateway.minute_bars("SPY", start, local_now).tz_convert(zone)
    prior = bars[bars.index.date == prior_day].between_time("09:30", "16:00")
    today = bars[bars.index.date == local_now.date()]
    if prior.empty:
        raise DataError(f"no regular-session bars for prior session {prior_day}")
    return signal(today, float(prior.iloc[-1]["close"]), minimum_gap)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="read and record, but never submit")
    arguments = parser.parse_args()

    base = load()
    # This is in memory only.  config/convex.yaml stays empty so the legacy
    # runner cannot create generic debit-vertical risk.
    config = base.with_overrides(
        {
            "structures.enabled": ["debit_vertical"],
            "structures.tie_break_order": ["debit_vertical"],
        }
    )
    live = AlpacaGateway(base)
    ledger = Ledger(base.path_("paths.ledger"))
    now, _ = live.clock()
    local_now = now.astimezone(ZoneInfo("America/New_York"))
    # The service starts at 10:00. Wait only for the completed 10:00 minute;
    # a later invocation is refused by observed_signal rather than traded late.
    if time(10, 0) <= local_now.time() < time(10, 1):
        sleep((datetime.combine(local_now.date(), time(10, 1), tzinfo=local_now.tzinfo) - local_now).total_seconds())
        now, _ = live.clock()
    try:
        found = observed_signal(live, now, minimum_gap=0.003)
    except ConvexError as error:
        ledger.append(
            Record(
                action=Action.STAND_DOWN,
                cycle_id=new_cycle_id(),
                structure="debit_vertical",
                rationale=f"Gap-continuation profile stood down: {error}",
                reject_reason="gap_signal_unavailable",
                extra={"profile": "gap_continuation_vertical"},
            )
        )
        print(f"gap profile stood down: {error}")
        return 0
    if found is None:
        ledger.append(
            Record(
                action=Action.STAND_DOWN,
                cycle_id=new_cycle_id(),
                structure="debit_vertical",
                rationale="Gap-continuation profile stood down: no qualifying overnight gap "
                "remained on the same side of VWAP at 10:00 ET.",
                reject_reason="gap_signal_absent",
                extra={"profile": "gap_continuation_vertical"},
            )
        )
        print("gap profile stood down: no qualifying signal")
        return 0

    expected_prefix = "bull call" if found.direction > 0 else "bear put"
    scenarios = build_scenarios(live, config)
    save_scenarios(scenarios, config.path_("paths.scenario_archive"))
    agent = Agent(
        gateway=DryRunGateway(live) if arguments.dry_run else live,
        config=config,
        ledger=ledger,
        scenarios=scenarios,
        submission_cutoff=SUBMISSION_CUTOFF,
        dry_run=arguments.dry_run,
        candidate_filter=lambda candidate: candidate.description.startswith(expected_prefix),
        receipt_context=found.as_dict(),
        reprice_ticks=(),
        decision_probability=PROFILE_PROBABILITY,
        decision_source=PROFILE_SOURCE,
    )
    result = agent.run_cycle(
        prior_returns=scenarios.log_returns.tolist(),
        variance_history=[],
        family_pnl=family_results(ledger),
    )
    print(f"cycle {result.cycle_id}: {result.reason}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
