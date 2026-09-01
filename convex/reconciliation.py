"""Append-only reconciliation of broker fills that older receipts mislabelled.

This command is intentionally narrow. It reads the historical parent/child
orders and account activities through Alpaca MCP, validates that every fill can
be mapped to a recorded structure, then appends corrections. It never rewrites
a receipt and it refuses allocation when an activity cannot be accounted for.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable

from convex.errors import DataError
from convex.ledger import Action, Ledger, Record


CENT = Decimal("0.01")
MULTIPLIER = Decimal("100")


@dataclass(frozen=True)
class Fill:
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: Decimal


@dataclass(frozen=True)
class Entry:
    source: dict[str, Any]
    submitted: dict[str, Any]
    parent: dict[str, Any]
    legs: tuple[Fill, ...]


def _value(raw: dict[str, Any], key: str, context: str) -> Any:
    value = raw.get(key)
    if value is None or value == "":
        raise DataError(f"{context}: missing {key!r}")
    return value


def _decimal(raw: dict[str, Any], key: str, context: str) -> Decimal:
    try:
        return Decimal(str(_value(raw, key, context)))
    except Exception as error:
        raise DataError(f"{context}: {key!r} is not a decimal") from error


def _quantity(raw: dict[str, Any], key: str, context: str) -> int:
    value = _decimal(raw, key, context)
    if value <= 0 or value != value.to_integral_value():
        raise DataError(f"{context}: {key}={value} is not a positive whole quantity")
    return int(value)


def _full_fill(raw: dict[str, Any], context: str) -> Fill:
    if str(_value(raw, "status", context)).lower() != "filled":
        raise DataError(f"{context}: broker status is {raw.get('status')!r}, not filled")
    qty = _quantity(raw, "qty", context)
    filled = _quantity(raw, "filled_qty", context)
    if filled != qty:
        raise DataError(f"{context}: partial fill {filled}/{qty} cannot be reconciled as closed")
    return Fill(
        order_id=str(_value(raw, "id", context)),
        # Parent multi-leg orders report empty symbol and side. Their child
        # receipts are validated separately before any cash is calculated.
        symbol=str(raw.get("symbol") or ""),
        side=str(raw.get("side") or "").lower(),
        quantity=filled,
        price=_decimal(raw, "filled_avg_price", context),
    )


def _cash(fill: Fill) -> Decimal:
    if fill.side in {"sell", "sell_short"}:
        return fill.price * fill.quantity * MULTIPLIER
    if fill.side == "buy":
        return -fill.price * fill.quantity * MULTIPLIER
    raise DataError(f"{fill.order_id}: unsupported broker fill side {fill.side!r}")


def _entry_date(record: dict[str, Any]) -> date:
    stamp = str(_value(record, "ts", "entry receipt"))
    try:
        return date.fromisoformat(stamp[:10])
    except ValueError as error:
        raise DataError(f"entry receipt has invalid timestamp {stamp!r}") from error


def _leg_fills(parent: dict[str, Any], context: str) -> tuple[Fill, ...]:
    raw_legs = parent.get("legs")
    if not isinstance(raw_legs, list) or not raw_legs:
        raise DataError(f"{context}: parent order carries no child leg receipts")
    return tuple(_full_fill(leg, f"{context} child {index}") for index, leg in enumerate(raw_legs))


def _entry_sources(records: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    submissions: dict[tuple[str, str], dict[str, Any]] = {}
    corrected: set[str] = set()
    for record in records:
        action = record.get("action")
        if action == Action.ORDER_SUBMITTED.value:
            submissions[(str(record.get("cycle_id")), str(record.get("structure")))] = record
        if action == Action.ORDER_RECONCILED.value:
            order_id = str(record.get("entry_order_id", ""))
            if order_id:
                corrected.add(order_id)

    sources: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in records:
        if record.get("action") != Action.ORDER_FILLED.value:
            continue
        order_id = str((record.get("outcome") or {}).get("order_id", ""))
        submitted = submissions.get((str(record.get("cycle_id")), str(record.get("structure"))))
        if order_id and submitted is not None and order_id not in corrected:
            sources.append((record, submitted))
    return sources


def _close_sources(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("action") == Action.POSITION_CLOSED.value
        and "realised_pnl" not in (record.get("outcome") or {})
        and (record.get("outcome") or {}).get("order_id")
    ]


def _round(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def reconcile(gateway, ledger: Ledger, *, write: bool = False) -> list[dict[str, Any]]:
    """Read broker receipts, validate all mappings, then optionally append them."""
    records = list(ledger.read())
    sources = _entry_sources(records)
    if not sources:
        return []

    entries: list[Entry] = []
    for source, submitted in sources:
        order_id = str((source.get("outcome") or {}).get("order_id"))
        parent = gateway.order_raw(order_id)
        _full_fill(parent, f"entry {order_id}")
        legs = _leg_fills(parent, f"entry {order_id}")
        expected = {
            str(leg["symbol"]): abs(int(leg["ratio"])) * int(submitted["contracts"])
            for leg in submitted.get("legs") or []
        }
        observed = defaultdict(int)
        for leg in legs:
            observed[leg.symbol] += leg.quantity
        if dict(observed) != expected:
            raise DataError(
                f"entry {order_id}: child fill quantities {dict(observed)} do not match "
                f"the submitted structure {expected}"
            )
        entries.append(Entry(source, submitted, parent, legs))

    close_fills: list[tuple[dict[str, Any], Fill]] = []
    for source in _close_sources(records):
        order_id = str((source.get("outcome") or {}).get("order_id"))
        close_fills.append((source, _full_fill(gateway.order_raw(order_id), f"close {order_id}")))

    close_fills.sort(key=lambda pair: str(pair[0].get("ts", "")))
    remaining = [
        {"source": source, "fill": fill, "left": fill.quantity}
        for source, fill in close_fills
    ]

    allocations: dict[str, list[Fill]] = {}
    for entry in entries:
        allocated: list[Fill] = []
        for leg in entry.legs:
            needed = leg.quantity
            expected_close_side = "sell" if leg.side == "buy" else "buy"
            for close in remaining:
                fill = close["fill"]
                if needed == 0:
                    break
                if fill.symbol != leg.symbol or fill.side != expected_close_side or close["left"] == 0:
                    continue
                quantity = min(needed, close["left"])
                allocated.append(Fill(fill.order_id, fill.symbol, fill.side, quantity, fill.price))
                needed -= quantity
                close["left"] -= quantity
            if needed:
                raise DataError(
                    f"{entry.parent['id']}: {leg.symbol} has {needed} contracts without a "
                    "verified matching close"
                )
        allocations[str(entry.parent["id"])] = allocated

    unmatched = [item for item in remaining if item["left"]]
    if unmatched:
        raise DataError(
            "broker closes could not be assigned to a recorded entry: "
            + ", ".join(f"{item['fill'].symbol} x{item['left']}" for item in unmatched)
        )

    by_day: dict[date, list[Entry]] = defaultdict(list)
    for entry in entries:
        by_day[_entry_date(entry.source)].append(entry)

    results: list[dict[str, Any]] = []
    for day, day_entries in sorted(by_day.items()):
        activities = gateway.account_activities(
            after=day, before=day + timedelta(days=1), activity_types=""
        )
        fee_total = -sum(
            _decimal(activity, "net_amount", "fee activity")
            for activity in activities
            if str(activity.get("activity_type", "")).upper() == "FEE"
        )
        activity_quantity = sum(
            _quantity(activity, "qty", "fill activity")
            for activity in activities
            if str(activity.get("activity_type", "")).upper() == "FILL"
        )

        traded_quantity = sum(
            sum(leg.quantity for leg in entry.legs)
            + sum(leg.quantity for leg in allocations[str(entry.parent["id"])])
            for entry in day_entries
        )
        if activity_quantity != traded_quantity:
            raise DataError(
                f"{day}: activity fills total {activity_quantity} contracts but recorded structures "
                f"total {traded_quantity}; fees cannot be allocated honestly"
            )

        allocated_fees: list[Decimal] = []
        for index, entry in enumerate(day_entries):
            units = sum(leg.quantity for leg in entry.legs) + sum(
                leg.quantity for leg in allocations[str(entry.parent["id"])]
            )
            if index == len(day_entries) - 1:
                fee = fee_total - sum(allocated_fees)
            else:
                fee = _round(fee_total * Decimal(units) / Decimal(traded_quantity))
            allocated_fees.append(fee)

        for entry, fee in zip(day_entries, allocated_fees):
            exit_legs = allocations[str(entry.parent["id"])]
            entry_cash = sum((_cash(leg) for leg in entry.legs), Decimal())
            exit_cash = sum((_cash(leg) for leg in exit_legs), Decimal())
            gross = entry_cash + exit_cash
            net = gross - fee
            result = {
                "structure": str(entry.source.get("structure")),
                "entry_order_id": str(entry.parent["id"]),
                "entry_cash": float(_round(entry_cash)),
                "exit_cash": float(_round(exit_cash)),
                "broker_fees": float(_round(fee)),
                "gross_realised_pnl": float(_round(gross)),
                "realised_pnl": float(_round(net)),
                "entry_legs": len(entry.legs),
                "exit_legs": len(exit_legs),
            }
            results.append(result)
            if not write:
                continue

            parent = _full_fill(entry.parent, f"entry {entry.parent['id']}")
            ledger.append(
                Record(
                    action=Action.ORDER_RECONCILED,
                    cycle_id=str(entry.source["cycle_id"]),
                    structure=str(entry.source.get("structure")),
                    rationale=(
                        f"Broker reconciliation verified entry {parent.order_id} fully filled "
                        f"for {parent.quantity} contracts. This corrects an earlier "
                        "pending_new receipt without rewriting it."
                    ),
                    legs=entry.submitted.get("legs"),
                    net_price=float(parent.price),
                    cost_breakdown=entry.submitted.get("cost_breakdown"),
                    contracts=int(entry.submitted["contracts"]),
                    outcome={
                        "order_id": parent.order_id,
                        "status": "filled",
                        "submitted_at": str(_value(entry.parent, "submitted_at", "entry")),
                        "client_order_id": str(entry.parent.get("client_order_id", "")),
                        "filled_qty": str(parent.quantity),
                        "filled_avg_price": str(parent.price),
                    },
                    extra={
                        "entry_order_id": parent.order_id,
                        "reconciles_sequence": entry.source.get("seq"),
                        "broker_leg_fills": [
                            {
                                "order_id": leg.order_id,
                                "symbol": leg.symbol,
                                "side": leg.side,
                                "qty": leg.quantity,
                                "price": str(leg.price),
                            }
                            for leg in entry.legs
                        ],
                    },
                )
            )
            ledger.append(
                Record(
                    action=Action.POSITION_RECONCILED,
                    cycle_id=str(entry.source["cycle_id"]),
                    structure=str(entry.source.get("structure")),
                    rationale=(
                        f"Broker fills reconcile {entry.source.get('structure')} to "
                        f"{float(_round(net)):+.2f} after {float(_round(fee)):.2f} in "
                        "day-level broker fees allocated by executed contracts."
                    ),
                    contracts=int(entry.submitted["contracts"]),
                    outcome={
                        "realised_pnl": float(_round(net)),
                        "gross_realised_pnl": float(_round(gross)),
                        "execution_cost": float(_round(fee)),
                        "broker_fees": float(_round(fee)),
                        "entry_cash": float(_round(entry_cash)),
                        "exit_cash": float(_round(exit_cash)),
                        "entry_order_id": str(entry.parent["id"]),
                        "exit_order_ids": [leg.order_id for leg in exit_legs],
                        "basis": "broker_fills_with_day_level_fee_allocation",
                    },
                    extra={
                        "entry_order_id": str(entry.parent["id"]),
                        "fee_allocation": "proportional_to_verified_executed_contracts",
                        "broker_fee_total_for_day": float(_round(fee_total)),
                    },
                )
            )

    return results
