"""The position manager.

The research holds its structures unhedged to expiration, so this module is
deliberately not a trade manager. It does not take profits, it does not stop
losses, and it does not adjust. Stopping out of a defined-risk structure turns
a capped loss into a realised loss plus four legs of slippage, which is the
same friction the whole project exists to avoid paying.

Exactly three things end a position early, and nothing else can:

  the kill switch          an operator has said stop, so nothing stays open
  the daily loss limit     the account is down its budget for the day
  the assignment guard     SPY settles into shares, and shares are not a
                           defined-risk position

The guard is the one that is specific to this project. The research is SPXW:
cash settled, European, no assignment. SPY is physically settled and American,
so a leg left open into the close can turn into a hundred shares of stock per
contract overnight. That is not a small difference in a footnote; it is the
single largest structural risk introduced by moving the paper's protocol onto
an instrument Alpaca actually lists.

Every early exit is an atomic multi-leg close. A leg-by-leg fallback is
forbidden: it can turn a defined-risk payoff into a different exposure while
later orders are rejected or delayed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from datetime import datetime
from enum import StrEnum
from typing import Any, Sequence

from convex.config import Config
from convex.errors import ConvexError, DataError, ExecutionError
from convex.execution import outcome_fields, resolve_order
from convex.instruments import OptionContract, Quote, parse_occ_symbol
from convex.ledger import Action, Ledger, Record, new_cycle_id


class Trigger(StrEnum):
    """Why a position is being closed before it would have expired."""

    KILL_SWITCH = "kill_switch"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    ASSIGNMENT_GUARD = "assignment_guard"


@dataclass(frozen=True)
class OpenLeg:
    """One option position, as the account reports it."""

    contract: OptionContract
    contracts: int  # signed: positive is long, negative is short
    average_entry_price: float
    unrealised_pnl: float

    @property
    def symbol(self) -> str:
        return self.contract.symbol

    @property
    def is_short(self) -> bool:
        return self.contracts < 0

    def settles_into_shares(self, spot: float, pin_band: float) -> bool:
        """Whether leaving this leg alone risks a share position tomorrow.

        In the money is the obvious case. The band around the strike is the
        less obvious one: a leg a few cents out of the money at 15:45 is one
        ordinary minute away from being exercised, and the guard has to decide
        before it knows where the close lands.
        """
        if pin_band < 0.0:
            raise DataError(f"pin band must not be negative, found {pin_band}")
        return abs(spot - self.contract.strike) <= pin_band or self.contract.is_itm(spot)


@dataclass(frozen=True)
class ClosePlan:
    """One leg the manager intends to close, and the price it will pay."""

    leg: OpenLeg
    limit_price: float
    quote: Quote

    @property
    def crosses_for(self) -> float:
        """Per-share distance from mid to the limit: the cost of getting out."""
        return abs(self.limit_price - self.quote.mid)


@dataclass(frozen=True)
class StructureClosePlan:
    """A complete, broker-attributable structure to close in one order."""

    entry_cycle_id: str
    structure: str
    legs: tuple[ClosePlan, ...]

    @property
    def contracts(self) -> int:
        quantities = [abs(plan.leg.contracts) for plan in self.legs]
        if not 2 <= len(quantities) <= 4 or any(quantity == 0 for quantity in quantities):
            raise ExecutionError("an atomic close needs two to four non-zero legs")
        size = quantities[0]
        for quantity in quantities[1:]:
            size = gcd(size, quantity)
        if size <= 0 or any(quantity % size for quantity in quantities):
            raise ExecutionError("open quantities do not form an integral structure")
        return size

    @property
    def limit_price(self) -> float:
        """Net debit to exit at the displayed touches; negative is a credit."""
        size = self.contracts
        return sum(
            (-plan.limit_price if plan.leg.contracts > 0 else plan.limit_price)
            * (abs(plan.leg.contracts) / size)
            for plan in self.legs
        )


@dataclass
class ManagerReport:
    """What one review pass did."""

    cycle_id: str
    triggers: list[Trigger] = field(default_factory=list)
    closed: list[dict] = field(default_factory=list)
    cancelled: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    left_open: list[str] = field(default_factory=list)
    reason: str = "nothing to do"

    @property
    def acted(self) -> bool:
        return bool(self.closed or self.cancelled or self.failed)


def read_positions(raw_positions: Sequence, underlying: str, multiplier: int) -> list[OpenLeg]:
    """Turn Alpaca's position records into legs this module can reason about.

    Only options on the configured underlying are returned. A share position is
    not silently ignored: it means an assignment already happened, and the
    caller is told about it rather than left to discover it in the equity curve.
    """
    legs: list[OpenLeg] = []
    shares: list[str] = []
    for position in raw_positions:
        symbol = str(position.symbol)
        asset_class = str(getattr(position, "asset_class", ""))
        if "option" not in asset_class.lower():
            if symbol == underlying:
                shares.append(symbol)
            continue
        contract = parse_occ_symbol(symbol, multiplier)
        if contract.underlying != underlying:
            continue
        legs.append(
            OpenLeg(
                contract=contract,
                contracts=int(float(position.qty)),
                average_entry_price=float(position.avg_entry_price),
                unrealised_pnl=float(position.unrealized_pl),
            )
        )
    if shares:
        raise ExecutionError(
            f"the account holds a {underlying} share position ({', '.join(shares)}); an "
            "assignment has already occurred and no automated close is safe until an "
            "operator has looked at it"
        )
    return legs


def closing_limit(leg: OpenLeg, quote: Quote) -> float:
    """The touch on the side the order has to cross to get out.

    A long is sold into the bid and a short is bought from the ask. This is a
    marketable limit rather than a market order: the price is bounded by a
    quote that was measured a moment ago, so a book that empties between the
    decision and the fill costs an unfilled order, not an arbitrary one.
    """
    price = quote.bid if leg.contracts > 0 else quote.ask
    if price <= 0.0:
        raise DataError(
            f"{leg.symbol}: no {'bid' if leg.contracts > 0 else 'ask'} to close against "
            f"(bid={quote.bid} ask={quote.ask})"
        )
    return price


def legs_to_close(
    legs: Sequence[OpenLeg],
    triggers: Sequence[Trigger],
    spot: float,
    pin_band: float,
) -> list[OpenLeg]:
    """Which legs make their entire attributed structure require an atomic exit.

    The kill switch and loss limit select every leg. The assignment guard
    selects the legs that could settle into shares; the manager then expands
    that signal to the complete, broker-attributed multi-leg structure rather
    than sending any selected leg by itself.
    """
    if not triggers:
        return []
    flatten = any(
        trigger in (Trigger.KILL_SWITCH, Trigger.DAILY_LOSS_LIMIT) for trigger in triggers
    )
    selected = [
        leg for leg in legs if flatten or leg.settles_into_shares(spot, pin_band)
    ]
    # This is a trigger set, not an execution sequence. The manager sends the
    # complete structure in one broker order.
    return sorted(selected, key=lambda leg: leg.symbol)


class PositionManager:
    """Watches what is open and closes it only for the three reasons above."""

    def __init__(self, gateway, config: Config, ledger: Ledger) -> None:
        self.gateway = gateway
        self.config = config
        self.ledger = ledger
        self.symbol = config.str_("underlying.symbol")
        self.multiplier = config.int_("underlying.contract_multiplier")

    # ----------------------------------------------------------------- triggers

    def _triggers(self, now: datetime, session_close: datetime, day_pnl_pct: float) -> list[Trigger]:
        found: list[Trigger] = []
        if self.config.path_("paths.kill_switch").exists():
            found.append(Trigger.KILL_SWITCH)

        limit = self.config.float_("risk.daily_loss_limit_pct")
        if day_pnl_pct <= -limit:
            found.append(Trigger.DAILY_LOSS_LIMIT)

        guard = self.config.float_("session.assignment_guard_minutes")
        minutes_left = (session_close - now).total_seconds() / 60.0
        if minutes_left <= guard:
            found.append(Trigger.ASSIGNMENT_GUARD)
        return found

    # ------------------------------------------------------------------- review

    def _cancel_working_orders(self, report: ManagerReport) -> None:
        """Pull any order still resting once a closing trigger has fired.

        An unfilled limit order is a position the account has not taken yet,
        and one that fills inside the assignment window opens exactly what the
        window exists to empty. Nothing else in the system would notice: the
        cycle sends an order and never looks at it again, and the guard reads
        positions, which a working order is not.

        This runs before the positions are read, so it happens whether or not
        anything is currently open.
        """
        for order in self.gateway.open_orders():
            try:
                self.gateway.cancel_order(order.id)
            except ConvexError as error:
                report.failed.append({"symbol": order.client_order_id, "error": str(error)})
                continue
            report.cancelled.append(
                {"order_id": order.id, "client_order_id": order.client_order_id}
            )
        if report.cancelled:
            self.ledger.append(
                Record(
                    action=Action.ORDER_REJECTED,
                    cycle_id=report.cycle_id,
                    rationale=(
                        f"Cancelled {len(report.cancelled)} working order(s) on "
                        + ", ".join(str(trigger) for trigger in report.triggers)
                        + ". An order that fills inside the guard window opens a "
                        "position the window exists to empty."
                    ),
                    reject_reason=str(report.triggers[0]),
                    extra={"cancelled": report.cancelled},
                )
            )

    def _verified_entry_for(self, legs: Sequence[OpenLeg]) -> dict[str, Any]:
        """Match the whole broker position to one verified entry receipt.

        This manager is deliberately limited to one concurrent structure. If
        the position cannot be matched exactly, it stops and asks for review
        rather than guessing which legs may safely be combined or closed.
        """
        actual = {leg.symbol: leg.contracts for leg in legs}
        for record in reversed(list(self.ledger.read())):
            if record.get("action") != Action.ORDER_FILLED.value or not record.get("legs"):
                continue
            outcome = record.get("outcome") or {}
            try:
                contracts = int(record["contracts"])
                filled = int(float(outcome.get("filled_qty", 0)))
            except (KeyError, TypeError, ValueError):
                continue
            if (
                contracts <= 0
                or filled != contracts
                or str(outcome.get("status", "")).lower() != "filled"
            ):
                continue
            expected = {
                str(item["symbol"]): int(item["ratio"]) * contracts
                for item in record["legs"]
            }
            if expected == actual:
                return record
        raise ExecutionError(
            "open option legs do not exactly match one broker-verified entry; "
            "refusing a leg-by-leg close"
        )

    def review(self, now: datetime, session_close: datetime) -> ManagerReport:
        """One pass: read the account, then atomically close an attributed structure."""
        report = ManagerReport(cycle_id=new_cycle_id())

        account = self.gateway.account()
        spot, _ = self.gateway.spot(self.symbol)
        report.triggers = self._triggers(now, session_close, account.day_pnl_pct)
        if report.triggers:
            self._cancel_working_orders(report)

        legs = read_positions(self.gateway.positions(), self.symbol, self.multiplier)
        if not legs:
            report.reason = (
                f"no open option positions, {len(report.cancelled)} order(s) cancelled"
                if report.cancelled
                else "no open option positions"
            )
            return report

        pin_band = spot * self.config.float_("session.pin_band_pct")
        targets = legs_to_close(legs, report.triggers, spot, pin_band)
        if not targets:
            report.left_open = [leg.symbol for leg in legs]
            report.reason = (
                f"{len(legs)} leg(s) open, held to expiry"
                if not report.triggers
                else f"{len(legs)} leg(s) open, none of them settle into shares at {spot:.2f}"
            )
            return report

        reason = ", ".join(str(trigger) for trigger in report.triggers)
        try:
            entry = self._verified_entry_for(legs)
            quotes = self.gateway.option_quotes([leg.symbol for leg in legs])
            max_age = self.config.float_("liquidity.max_quote_age_seconds")
            plan = StructureClosePlan(
                entry_cycle_id=str(entry["cycle_id"]),
                structure=str(entry.get("structure") or "structure"),
                legs=tuple(
                    ClosePlan(
                        leg,
                        closing_limit(leg, quotes[leg.symbol].require_fresh(max_age, now)),
                        quotes[leg.symbol],
                    )
                    for leg in legs
                ),
            )
            # Validate before a receipt says the close was attempted.
            _ = plan.contracts
            _ = plan.limit_price
        except ConvexError as error:
            detail = f"{error}; the structure remains intact for operator review"
            self.ledger.append(
                Record(
                    action=Action.RISK_HALT,
                    cycle_id=report.cycle_id,
                    rationale=f"Could not atomically close {len(legs)} open leg(s) on {reason}: {detail}.",
                    reject_reason="atomic_close_unavailable",
                    extra={"targets": [leg.symbol for leg in targets]},
                )
            )
            report.failed.append({"symbol": self.symbol, "error": detail})
            report.left_open = [leg.symbol for leg in legs]
            report.reason = detail
            return report

        self.ledger.append(
            Record(
                action=Action.RISK_HALT,
                cycle_id=report.cycle_id,
                rationale=(
                    f"Atomically closing {len(plan.legs)} open leg(s) on {reason}. "
                    f"{self.symbol} is at {spot:.2f} with "
                    f"{(session_close - now).total_seconds() / 60.0:.0f} minutes to the close."
                ),
                reject_reason=str(report.triggers[0]),
                extra={
                    "spot": round(spot, 4),
                    "pin_band": round(pin_band, 4),
                    "day_pnl_pct": round(account.day_pnl_pct, 6),
                    "targets": [item.leg.symbol for item in plan.legs],
                    "entry_cycle_id": plan.entry_cycle_id,
                },
            )
        )
        self._close_structure(plan, report)
        report.reason = f"atomically closed {len(plan.legs)} leg(s) on {reason}"
        return report

    # ------------------------------------------------------------------ closing

    def _close_structure(self, plan: StructureClosePlan, report: ManagerReport) -> None:
        client_order_id = f"convex-close-{report.cycle_id}-{plan.structure}"[:48]
        limit = round(plan.limit_price, 2)
        positions = [(item.leg.symbol, item.leg.contracts) for item in plan.legs]
        self.ledger.append(
            Record(
                action=Action.POSITION_CLOSE_SUBMITTED,
                cycle_id=report.cycle_id,
                structure=plan.structure,
                rationale=(
                    f"Submitting one atomic {len(plan.legs)}-leg close at {limit:.2f} "
                    f"for {plan.contracts} structure lot(s)."
                ),
                contracts=plan.contracts,
                net_price=limit,
                extra={"client_order_id": client_order_id, "entry_cycle_id": plan.entry_cycle_id},
            )
        )
        try:
            order = self.gateway.close_structure(positions, limit, client_order_id)
            resolution = resolve_order(
                self.gateway,
                order,
                plan.contracts,
                timeout_seconds=self.config.float_("execution.order_status_timeout_seconds"),
                poll_seconds=self.config.float_("execution.order_poll_seconds"),
            )
        except ConvexError as error:
            self.ledger.append(
                Record(
                    action=Action.ORDER_REJECTED,
                    cycle_id=report.cycle_id,
                    structure=plan.structure,
                    rationale=f"Could not atomically close {plan.structure}: {error}",
                    contracts=plan.contracts,
                    reject_reason="atomic_close_rejected",
                    extra={"client_order_id": client_order_id, "entry_cycle_id": plan.entry_cycle_id},
                )
            )
            report.failed.append({"symbol": plan.structure, "error": str(error)})
            return

        outcome = outcome_fields(resolution.order, cancel_requested=resolution.cancel_requested)
        outcome.update({"trigger": str(report.triggers[0])})
        if not resolution.filled:
            pending = not resolution.terminal
            self.ledger.append(
                Record(
                    action=Action.POSITION_CLOSE_PENDING if pending else Action.ORDER_REJECTED,
                    cycle_id=report.cycle_id,
                    structure=plan.structure,
                    rationale=(
                        f"Atomic close {resolution.order.id} is {resolution.order.status} with "
                        f"{resolution.order.filled_qty}/{plan.contracts} structures filled; "
                        "no leg-by-leg fallback is permitted."
                    ),
                    contracts=plan.contracts,
                    net_price=limit,
                    reject_reason="atomic_close_pending" if pending else "atomic_close_not_filled",
                    outcome=outcome,
                    extra={"client_order_id": client_order_id, "entry_cycle_id": plan.entry_cycle_id},
                )
            )
            report.failed.append({"symbol": plan.structure, "error": "atomic close was not fully filled"})
            return

        self.ledger.append(
            Record(
                action=Action.POSITION_CLOSED,
                cycle_id=report.cycle_id,
                structure=plan.structure,
                rationale=(
                    f"Alpaca verified atomic close {resolution.order.id} fully filled for "
                    f"{plan.contracts} structure lot(s)."
                ),
                contracts=plan.contracts,
                net_price=limit,
                outcome=outcome,
                extra={"client_order_id": client_order_id, "entry_cycle_id": plan.entry_cycle_id},
            )
        )
        report.closed.append({"structure": plan.structure, "order_id": str(resolution.order.id)})

    # --------------------------------------------------------------- settlement

    def settle(self, session_date, settlement_price: float) -> list[dict]:
        """Record what each structure opened that day was worth at expiry.

        This is what closes the loop the classifier feeds on: a family's own
        past results are one of its features, and they do not exist until a
        session has been settled. Structures the guard closed early are not
        settled here, because their result is a fill price rather than a payoff
        function, and inventing one from the payoff would be a receipt for a
        trade that did not happen. They are returned as unsettled instead.
        """
        opened: dict[str, dict] = {}
        settled_already: set[str] = set()
        closed_symbols: set[str] = set()

        for record in self.ledger.read():
            action = record.get("action")
            outcome = record.get("outcome") or {}
            if action == Action.ORDER_FILLED.value and record.get("legs"):
                try:
                    filled = int(float(outcome.get("filled_qty", 0)))
                    contracts = int(record["contracts"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    str(record.get("ts", ""))[:10] == str(session_date)
                    and str(outcome.get("status", "")).lower() == "filled"
                    and filled == contracts > 0
                ):
                    opened[f"{record['cycle_id']}/{record.get('structure')}"] = record
            elif action == Action.POSITION_CLOSED.value:
                if "realised_pnl" in outcome:
                    settled_already.add(f"{record['cycle_id']}/{record.get('structure')}")
                elif record.get("entry_cycle_id"):
                    settled_already.add(
                        f"{record['entry_cycle_id']}/{record.get('structure')}"
                    )
                elif record.get("structure"):
                    closed_symbols.add(str(record["structure"]))

        results: list[dict] = []
        for key, record in opened.items():
            if key in settled_already:
                continue
            legs = [_leg_from_record(entry, self.multiplier) for entry in record["legs"]]
            if any(contract.symbol in closed_symbols for contract, _ in legs):
                results.append(
                    {"structure": record.get("structure"), "cycle_id": record["cycle_id"],
                     "unsettled": "the guard closed at least one leg before expiry"}
                )
                continue

            contracts = int(record["contracts"])
            pnl = settlement_pnl(
                legs,
                float(record["net_price"]),
                contracts,
                settlement_price,
                self.multiplier,
            )
            self.ledger.append(
                Record(
                    action=Action.POSITION_CLOSED,
                    cycle_id=record["cycle_id"],
                    structure=record.get("structure"),
                    rationale=(
                        f"Expired with {self.symbol} settling at {settlement_price:.2f}. "
                        f"{contracts} lot(s) entered at {float(record['net_price']):.2f} "
                        f"were worth {pnl:+,.2f} dollars."
                    ),
                    contracts=contracts,
                    outcome={
                        "realised_pnl": round(pnl, 2),
                        "settlement_price": round(settlement_price, 4),
                        "basis": "payoff at the official close, held to expiry",
                        "session_date": str(session_date),
                    },
                )
            )
            results.append(
                {"structure": record.get("structure"), "cycle_id": record["cycle_id"],
                 "realised_pnl": round(pnl, 2)}
            )
        return results


def _leg_from_record(entry: dict, multiplier: int) -> tuple[OptionContract, int]:
    """Rebuild one leg of a recorded structure from its ledger line."""
    for key in ("symbol", "ratio"):
        if key not in entry:
            raise DataError(f"ledger leg is missing {key!r}: {entry}")
    return parse_occ_symbol(str(entry["symbol"]), multiplier), int(entry["ratio"])


def settlement_pnl(
    legs: Sequence[tuple[OptionContract, int]],
    net_entry_debit: float,
    contracts: int,
    settlement_price: float,
    multiplier: int,
) -> float:
    """What a structure held to expiry was actually worth, in dollars.

    Every 0DTE structure this agent opens either expires or is closed by the
    guard, so for the ones that expire the result is not an estimate: it is the
    payoff function evaluated at the settlement price, which is arithmetic with
    one input. That input is the official close, read from the market data API
    rather than from the last quote the agent happened to see.
    """
    if contracts <= 0:
        raise DataError(f"cannot settle a structure of {contracts} contracts")
    if settlement_price <= 0.0:
        raise DataError(f"settlement price must be positive, found {settlement_price}")
    intrinsic = sum(
        ratio * contract.intrinsic(settlement_price) for contract, ratio in legs
    )
    return (intrinsic - net_entry_debit) * multiplier * contracts
