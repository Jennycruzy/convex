"""Broker-order state resolution.

Submitting an order asks Alpaca to trade. It is not evidence that Alpaca did
trade. Both the entry agent and the assignment guard use this module so they
apply the same rule: a receipt may say *filled* only after the broker reports a
full fill for the requested quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from convex.data.alpaca import OrderRecord
from convex.errors import DataError


TERMINAL_STATUSES = frozenset(
    {
        "filled", "canceled", "cancelled", "rejected", "expired", "done_for_day",
        "stopped", "suspended", "calculated", "replaced",
    }
)


def filled_quantity(order: OrderRecord) -> int:
    """Return a whole option quantity, never a guessed zero."""
    try:
        quantity = Decimal(order.filled_qty)
    except (InvalidOperation, ValueError) as error:
        raise DataError(f"order {order.id}: filled_qty={order.filled_qty!r} is not numeric") from error
    if quantity < 0 or quantity != quantity.to_integral_value():
        raise DataError(
            f"order {order.id}: filled_qty={order.filled_qty!r} is not a whole non-negative quantity"
        )
    return int(quantity)


def is_full_fill(order: OrderRecord, requested_contracts: int) -> bool:
    if requested_contracts <= 0:
        raise DataError(f"requested order quantity must be positive, found {requested_contracts}")
    return order.status.lower() == "filled" and filled_quantity(order) == requested_contracts


def outcome_fields(order: OrderRecord, *, cancel_requested: bool = False) -> dict[str, str | bool | None]:
    """The factual broker fields every order receipt must preserve."""
    return {
        "order_id": order.id,
        "status": order.status,
        "submitted_at": order.submitted_at,
        "client_order_id": order.client_order_id,
        "filled_qty": order.filled_qty,
        "filled_avg_price": order.filled_avg_price,
        "cancel_requested": cancel_requested,
    }


@dataclass(frozen=True)
class OrderResolution:
    order: OrderRecord
    cancel_requested: bool
    requested_contracts: int

    @property
    def filled(self) -> bool:
        return is_full_fill(self.order, self.requested_contracts)

    @property
    def terminal(self) -> bool:
        return self.order.status.lower() in TERMINAL_STATUSES


def resolve_order(gateway, submitted: OrderRecord, requested_contracts: int, *, timeout_seconds: float,
                  poll_seconds: float) -> OrderResolution:
    """Poll a submission and cancel it if it remains working at the deadline."""
    latest = gateway.wait_for_order(submitted.id, timeout_seconds, poll_seconds)
    cancel_requested = False
    if not is_full_fill(latest, requested_contracts) and latest.status.lower() not in TERMINAL_STATUSES:
        gateway.cancel_order(submitted.id)
        cancel_requested = True
        latest = gateway.wait_for_order(submitted.id, timeout_seconds, poll_seconds)
    return OrderResolution(latest, cancel_requested, requested_contracts)
