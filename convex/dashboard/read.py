"""Turning the decision ledger into what the page shows.

Nothing in this module invents anything. Every figure on the dashboard is read
back out of the append-only ledger the agent wrote at decision time, which is
the point of having written it: the page a judge loads and the evidence in the
write-up are the same artefact, and neither is assembled after the fact.

That has one visible consequence. Before the agent has run, this returns empty
summaries and the page says so plainly. It does not render a sample trade, a
placeholder equity curve, or a demo mode. A dashboard showing numbers the agent
never produced would be the exact failure the project exists to argue against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from convex.ledger import Action

# Actions that represent a decision about a specific structure, in the order a
# reader cares about them.
DECISION_ACTIONS = (
    Action.ORDER_SUBMITTED,
    Action.ORDER_FILLED,
    Action.ORDER_REJECTED,
    Action.CANDIDATE_REJECTED,
    Action.STAND_DOWN,
    Action.RISK_HALT,
    Action.POSITION_CLOSED,
)


def load(path: Path) -> list[dict[str, Any]]:
    """Every ledger line, oldest first. A malformed line is not skipped."""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number} is not valid JSON: {error}") from error
    return sorted(records, key=lambda record: record.get("seq", 0))


@dataclass(frozen=True)
class Summary:
    """The headline counts, all of them derived from receipts."""

    cycles: int = 0
    decisions: int = 0
    orders: int = 0
    refusals: int = 0
    stand_downs: int = 0
    halts: int = 0
    realised_pnl: float = 0.0
    settled_structures: int = 0
    execution_cost: float = 0.0
    first_seen: str | None = None
    last_seen: str | None = None

    @property
    def has_run(self) -> bool:
        return self.decisions > 0 or self.cycles > 0

    @property
    def refusal_rate(self) -> float:
        """Share of structure-level verdicts that were refusals.

        Published deliberately. A low number is not obviously good: the whole
        argument is that most 0DTE candidates do not survive their own
        execution cost, so an agent that never refuses is not being careful.
        """
        considered = self.orders + self.refusals
        return self.refusals / considered if considered else 0.0


def summarise(records: Iterable[dict[str, Any]]) -> Summary:
    rows = list(records)
    if not rows:
        return Summary()

    cycles = {row.get("cycle_id") for row in rows if row.get("cycle_id")}
    counts = {action: 0 for action in Action}
    realised = 0.0
    settled = 0
    cost = 0.0

    for row in rows:
        action = row.get("action")
        for candidate in Action:
            if action == candidate.value:
                counts[candidate] += 1
        outcome = row.get("outcome") or {}
        if action == Action.POSITION_CLOSED.value and "realised_pnl" in outcome:
            realised += float(outcome["realised_pnl"])
            settled += 1
        if action == Action.ORDER_SUBMITTED.value:
            breakdown = row.get("cost_breakdown")
            # Law 3. A submitted order without its cost breakdown or its lot
            # count is a ledger this page cannot total honestly, and the old
            # reading of it assumed one lot, which invented the number rather
            # than reporting that it was missing.
            if breakdown is None or row.get("contracts") is None:
                raise ValueError(
                    f"ledger row {row.get('seq')} submitted an order without "
                    "a cost breakdown and a contract count; the page cannot "
                    "total execution cost from it"
                )
            if "total" not in breakdown:
                raise ValueError(
                    f"ledger row {row.get('seq')} has a cost breakdown with no "
                    f"'total' (keys: {sorted(breakdown)})"
                )
            cost += float(breakdown["total"]) * int(row["contracts"])

    stamps = [row["ts"] for row in rows if row.get("ts")]
    decisions = sum(counts[action] for action in DECISION_ACTIONS)
    return Summary(
        cycles=len(cycles),
        decisions=decisions,
        orders=counts[Action.ORDER_SUBMITTED],
        refusals=counts[Action.CANDIDATE_REJECTED],
        stand_downs=counts[Action.STAND_DOWN],
        halts=counts[Action.RISK_HALT],
        realised_pnl=round(realised, 2),
        settled_structures=settled,
        execution_cost=round(cost, 2),
        first_seen=min(stamps) if stamps else None,
        last_seen=max(stamps) if stamps else None,
    )


@dataclass
class Cycle:
    """One decision pass, with everything it produced."""

    cycle_id: str
    started: str
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def snapshot(self) -> dict[str, Any] | None:
        for record in self.records:
            if record.get("action") == Action.SNAPSHOT.value:
                return record
        return None

    @property
    def features(self) -> dict[str, float]:
        snapshot = self.snapshot
        return (snapshot or {}).get("features") or {}

    @property
    def verdicts(self) -> list[dict[str, Any]]:
        """One row per structure this cycle reached a conclusion about."""
        wanted = {
            Action.ORDER_SUBMITTED.value,
            Action.CANDIDATE_REJECTED.value,
            Action.ORDER_REJECTED.value,
            Action.STAND_DOWN.value,
        }
        return [record for record in self.records if record.get("action") in wanted]

    @property
    def opened(self) -> int:
        return sum(
            1 for record in self.records if record.get("action") == Action.ORDER_SUBMITTED.value
        )

    @property
    def stood_down(self) -> bool:
        return self.opened == 0


def cycles(records: Iterable[dict[str, Any]]) -> list[Cycle]:
    """Group the ledger into cycles, most recent first."""
    grouped: dict[str, Cycle] = {}
    for record in records:
        cycle_id = record.get("cycle_id")
        if not cycle_id:
            continue
        cycle = grouped.get(cycle_id)
        if cycle is None:
            cycle = Cycle(cycle_id=cycle_id, started=record.get("ts", ""))
            grouped[cycle_id] = cycle
        cycle.records.append(record)
    return sorted(grouped.values(), key=lambda cycle: cycle.started, reverse=True)


def waterfalls(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every decision that carries a gross-to-net breakdown, newest first.

    Refusals come first within a timestamp because they are the interesting
    ones: a candidate with real gross edge that did not survive its own
    half-spread is the argument this project is making, and it is the visual
    that is hard to find anywhere else.
    """
    rows = [record for record in records if record.get("waterfall")]
    # Refusals sort ahead of everything, then by recency. A candidate with real
    # gross edge that did not survive its own half-spread is the finding this
    # project is built on, and a page that leads with a winning fill instead
    # buries it.
    rows.sort(
        key=lambda record: (
            record.get("action") == Action.CANDIDATE_REJECTED.value,
            record.get("ts", ""),
        ),
        reverse=True,
    )
    return rows


def realised_curve(records: Iterable[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """The account's own running result, gross and net, one point per close.

    Read from ``position_closed`` records in ledger order, which is settlement
    order. Nothing here is modelled or replayed: a point exists only where the
    agent opened a position and it was closed out for a number the ledger
    recorded.

    That means this returns two empty lists until the agent has actually
    traded, and the caller is expected to say so rather than draw a flat line
    at zero. A flat line reads as a strategy that traded and broke even, which
    is a different and much more flattering claim than having not traded.

    A close without ``realised_pnl`` is skipped and not counted, on the same
    principle as the cost totals above: a receipt missing its number is not a
    zero.
    """
    gross_running = 0.0
    net_running = 0.0
    gross_curve: list[float] = []
    net_curve: list[float] = []
    for row in records:
        if row.get("action") != Action.POSITION_CLOSED.value:
            continue
        outcome = row.get("outcome") or {}
        if "realised_pnl" not in outcome:
            continue
        net = float(outcome["realised_pnl"])
        # The realised figure is already net of what it cost to get in and out.
        # Gross is only available where the close recorded the cost alongside
        # it; where it did not, the two curves coincide and the chart shows no
        # band rather than an invented one.
        cost = float(outcome.get("execution_cost", 0.0))
        net_running += net
        gross_running += net + cost
        net_curve.append(round(net_running, 2))
        gross_curve.append(round(gross_running, 2))
    return gross_curve, net_curve


def find_decision(records: Iterable[dict[str, Any]], sequence: int) -> dict[str, Any] | None:
    for record in records:
        if record.get("seq") == sequence:
            return record
    return None


def format_stamp(value: str | None) -> str:
    """A ledger timestamp as a reader wants it, or a dash if there is none."""
    if not value:
        return "·"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    except ValueError:
        return value


def payoff_from_record(
    record: dict[str, Any], points: int = 161
) -> tuple[list[tuple[float, float]], tuple[float, ...]]:
    """Rebuild a structure's expiry payoff from its ledger line.

    The receipt carries every leg's right, strike and signed ratio plus the net
    entry price, which is all the payoff at expiry depends on. Nothing is
    fetched and nothing is modelled: this is the same arithmetic the risk check
    ran before the order was sent, recomputed from the record so a reader can
    see the shape the agent was looking at.

    Returns the curve and the strikes, so the caller can mark the kinks.
    """
    legs = record.get("legs") or []
    if not legs:
        raise ValueError("this record carries no legs to draw")
    net_entry = float(record.get("net_price", 0.0))
    multiplier = 100

    strikes: list[float] = []
    parsed: list[tuple[str, float, int]] = []
    for leg in legs:
        for key in ("right", "strike", "ratio"):
            if key not in leg:
                raise ValueError(f"a recorded leg is missing {key!r}")
        right = str(leg["right"]).lower()
        strike = float(leg["strike"])
        parsed.append((right, strike, int(leg["ratio"])))
        strikes.append(strike)

    # A window wide enough to show the flat tails on both sides of the wings,
    # which for a broken-wing butterfly is the whole point of the diagram.
    low, high = min(strikes), max(strikes)
    pad = max(high - low, high * 0.02)
    grid = {low - pad + (high - low + 2 * pad) * index / (points - 1) for index in range(points)}
    grid |= set(strikes)

    def intrinsic(right: str, strike: float, price: float) -> float:
        return max(price - strike, 0.0) if right == "call" else max(strike - price, 0.0)

    curve = [
        (
            price,
            (sum(ratio * intrinsic(right, strike, price) for right, strike, ratio in parsed)
             - net_entry)
            * multiplier
            * max(int(record.get("contracts") or 1), 1),
        )
        for price in sorted(grid)
    ]
    return curve, tuple(sorted(set(strikes)))
