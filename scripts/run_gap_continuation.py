"""Run the one-lot, deterministic 10:00 ET gap-continuation profile.

The production configuration remains entry-disabled.  This runner enables only
``debit_vertical`` in memory after the signal is observed, then lets the normal
Agent apply its full session and candidate-gate stack.  A canceled zero-fill
entry uses the configured fresh-quote retry ladder, and it cannot place more
than one contract.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, time, timedelta
from time import sleep
from zoneinfo import ZoneInfo

from convex.agent import Agent
from convex.config import Config, load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError, DataError
from convex.gap_continuation import GapContinuationSignal, signal
from convex.ledger import Action, Ledger, Record, new_cycle_id
from convex.scenarios import build as build_scenarios
from convex.scenarios import save as save_scenarios

SUBMISSION_CUTOFF = datetime(2026, 9, 4, 11, 0, tzinfo=ZoneInfo("America/New_York"))
PROFILE_SOURCE = (
    "gap-continuation profile: deterministic signal; admission score is not a "
    "calibrated probability"
)


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
            history[str(record["structure"])].append(float(outcome["realised_pnl"]))
    return dict(history)


def _configured_time(config: Config, key: str) -> time:
    raw = config.str_(key)
    try:
        return time.fromisoformat(raw)
    except ValueError as error:
        raise DataError(f"{key} must be an ISO time, found {raw!r}") from error


def observed_signal(
    gateway: AlpacaGateway, now: datetime, config: Config
) -> GapContinuationSignal | None:
    """Read the completed 10:00 bar and the preceding session close."""
    zone = ZoneInfo(config.str_("session.timezone"))
    signal_time = _configured_time(config, "session.entry_time")
    execution_start = _configured_time(config, "strategy.gap_continuation.execution_start")
    execution_end = _configured_time(config, "strategy.gap_continuation.execution_end")
    if execution_start > execution_end:
        raise DataError("gap profile execution_start must not be after execution_end")
    if now.tzinfo is None:
        raise DataError("exchange clock must carry a timezone")
    local_now = now.astimezone(zone)
    if not execution_start <= local_now.time() <= execution_end:
        raise DataError(
            f"gap profile only permits execution from "
            f"{execution_start.isoformat(timespec='minutes')} to "
            f"{execution_end.isoformat(timespec='minutes')} {zone.key}"
        )
    symbol = config.str_("underlying.symbol")
    sessions = gateway.sessions(local_now.date() - timedelta(days=10), local_now.date())
    dates = [session.session_date for session in sessions]
    previous = [day for day in dates if day < local_now.date()]
    if local_now.date() not in dates or not previous:
        raise DataError("no current and prior Alpaca trading session for gap signal")
    prior_day = previous[-1]
    start = datetime.combine(prior_day, time(9, 30), tzinfo=zone)
    bars = gateway.minute_bars(symbol, start, local_now).tz_convert(zone)
    close_time = _configured_time(config, "session.close_time")
    prior = bars[bars.index.date == prior_day].between_time(
        "09:30", close_time.isoformat(timespec="minutes")
    )
    today = bars[bars.index.date == local_now.date()]
    if prior.empty:
        raise DataError(f"no regular-session bars for prior session {prior_day}")
    return signal(
        today,
        float(prior.iloc[-1]["close"]),
        minimum_gap=config.float_("strategy.gap_continuation.minimum_gap"),
        signal_time=signal_time,
        minimum_vwap_distance=config.float_("strategy.gap_continuation.minimum_vwap_distance"),
    )


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
    zone = ZoneInfo(config.str_("session.timezone"))
    entry_time = _configured_time(config, "session.entry_time")
    execution_start = _configured_time(config, "strategy.gap_continuation.execution_start")
    local_now = now.astimezone(zone)
    # The service starts at the configured entry time. Wait only for the
    # completed signal minute; a later invocation is refused rather than traded late.
    if entry_time <= local_now.time() < execution_start:
        target = datetime.combine(local_now.date(), execution_start, tzinfo=zone)
        sleep((target - local_now).total_seconds())
        now, _ = live.clock()
    try:
        found = observed_signal(live, now, config)
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
        # Keep the active profile on the same configured fresh-quote retry
        # ladder as the generic runner. An empty ladder is an explicit policy;
        # this profile must not silently choose it by bypassing config.
        reprice_ticks=tuple(int(tick) for tick in config.list_("execution.reprice_ticks")),
        decision_probability=config.float_("strategy.gap_continuation.admission_score"),
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
