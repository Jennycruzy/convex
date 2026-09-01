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

Order of operations inside any flatten: shorts first, then longs. Closing a
short can only reduce the worst case. Closing a long first would leave a naked
short standing in the account for as long as the next order takes to fill, and
there is no window in which this agent is allowed to hold undefined risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Sequence

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
    """Which legs this pass closes, shorts first.

    The kill switch and the loss limit take everything. The assignment guard on
    its own takes only what would settle into shares, and leaves a long leg
    that is safely out of the money to expire worthless, because paying a
    half-spread to close something already worth nothing is a donation.
    """
    if not triggers:
        return []
    flatten = any(
        trigger in (Trigger.KILL_SWITCH, Trigger.DAILY_LOSS_LIMIT) for trigger in triggers
    )
    selected = [
        leg for leg in legs if flatten or leg.settles_into_shares(spot, pin_band)
    ]
    # Shorts first: closing one can only narrow the worst case, and no ordering
    # of these orders may ever leave a short standing without its wing.
    return sorted(selected, key=lambda leg: (not leg.is_short, leg.symbol))


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

    def review(self, now: datetime, session_close: datetime) -> ManagerReport:
        """One pass: read the account, decide, close what has to be closed."""
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
        report.left_open = [leg.symbol for leg in legs if leg not in targets]

        if not targets:
            report.reason = (
                f"{len(legs)} leg(s) open, held to expiry"
                if not report.triggers
                else f"{len(legs)} leg(s) open, none of them settle into shares at {spot:.2f}"
            )
            return report

        quotes = self.gateway.option_quotes([leg.symbol for leg in targets])
        max_age = self.config.float_("liquidity.max_quote_age_seconds")
        plans = [
            ClosePlan(leg, closing_limit(leg, quotes[leg.symbol].require_fresh(max_age, now)),
                      quotes[leg.symbol])
            for leg in targets
        ]

        reason = ", ".join(str(trigger) for trigger in report.triggers)
        self.ledger.append(
            Record(
                action=Action.RISK_HALT,
                cycle_id=report.cycle_id,
                rationale=(
                    f"Closing {len(plans)} of {len(legs)} open leg(s) on {reason}. "
                    f"{self.symbol} is at {spot:.2f} with "
                    f"{(session_close - now).total_seconds() / 60.0:.0f} minutes to the close."
                ),
                reject_reason=str(report.triggers[0]),
                extra={
                    "spot": round(spot, 4),
                    "pin_band": round(pin_band, 4),
                    "day_pnl_pct": round(account.day_pnl_pct, 6),
                    "targets": [plan.leg.symbol for plan in plans],
                },
            )
        )

        for plan in plans:
            self._close(plan, report)
        report.reason = f"closed {len(report.closed)} leg(s) on {reason}"
        return report

    # ------------------------------------------------------------------ closing

    def _close(self, plan: ClosePlan, report: ManagerReport) -> None:
        leg = plan.leg
        client_order_id = f"convex-close-{report.cycle_id}-{leg.symbol}"[:48]
        self.ledger.append(
            Record(
                action=Action.POSITION_CLOSE_SUBMITTED,
                cycle_id=report.cycle_id,
                structure=leg.symbol,
                rationale=(
                    f"Submitting a closing limit at {plan.limit_price:.2f} for "
                    f"{leg.contracts:+d} {leg.symbol}, crossing {plan.crosses_for:.2f} "
                    f"from a mid of {plan.quote.mid:.2f}."
                ),
                contracts=leg.contracts,
                net_price=round(plan.limit_price, 2),
                extra={"client_order_id": client_order_id},
            )
        )
        try:
            order = self.gateway.close_leg(
                leg.symbol, leg.contracts, plan.limit_price, client_order_id
            )
            resolution = resolve_order(
                self.gateway,
                order,
                abs(leg.contracts),
                timeout_seconds=self.config.float_("execution.order_status_timeout_seconds"),
                poll_seconds=self.config.float_("execution.order_poll_seconds"),
            )
        except ConvexError as error:
            self.ledger.append(
                Record(
                    action=Action.ORDER_REJECTED,
                    cycle_id=report.cycle_id,
                    structure=leg.symbol,
                    rationale=f"Could not close {leg.symbol}: {error}",
                    contracts=leg.contracts,
                    reject_reason="close_rejected",
                    extra={"client_order_id": client_order_id},
                )
            )
            report.failed.append({"symbol": leg.symbol, "error": str(error)})
            return

        outcome = outcome_fields(resolution.order, cancel_requested=resolution.cancel_requested)
        outcome.update(
            {
                "trigger": str(report.triggers[0]),
                "unrealised_pnl_at_close": round(leg.unrealised_pnl, 2),
                "average_entry_price": leg.average_entry_price,
            }
        )
        if not resolution.filled:
            pending = not resolution.terminal
            self.ledger.append(
                Record(
                    action=Action.POSITION_CLOSE_PENDING if pending else Action.ORDER_REJECTED,
                    cycle_id=report.cycle_id,
                    structure=leg.symbol,
                    rationale=(
                        f"Close {resolution.order.id} is {resolution.order.status} with "
                        f"{resolution.order.filled_qty}/{abs(leg.contracts)} contracts filled; "
                        "the position remains under guard."
                    ),
                    contracts=leg.contracts,
                    net_price=round(plan.limit_price, 2),
                    reject_reason="close_pending" if pending else "close_not_filled",
                    outcome=outcome,
                    extra={"client_order_id": client_order_id},
                )
            )
            report.failed.append(
                {
                    "symbol": leg.symbol,
                    "error": "closing order was not fully filled",
                    "order_id": resolution.order.id,
                }
            )
            return

        self.ledger.append(
            Record(
                action=Action.POSITION_CLOSED,
                cycle_id=report.cycle_id,
                structure=leg.symbol,
                rationale=(
                    f"Alpaca verified close {resolution.order.id} fully filled at "
                    f"{resolution.order.filled_avg_price} for {leg.contracts:+d} {leg.symbol}."
                ),
                contracts=leg.contracts,
                net_price=round(plan.limit_price, 2),
                outcome=outcome,
                extra={"client_order_id": client_order_id},
            )
        )
        report.closed.append({"symbol": leg.symbol, "order_id": str(resolution.order.id)})

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
            if action == Action.ORDER_SUBMITTED.value and record.get("legs"):
                if str(record.get("ts", ""))[:10] == str(session_date):
                    opened[f"{record['cycle_id']}/{record.get('structure')}"] = record
            elif action == Action.POSITION_CLOSED.value:
                outcome = record.get("outcome") or {}
                if "realised_pnl" in outcome:
                    settled_already.add(f"{record['cycle_id']}/{record.get('structure')}")
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
