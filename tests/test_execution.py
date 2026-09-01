"""Order receipts only become fills after the broker says they are full."""

from convex.data.alpaca import OrderRecord
from convex.execution import is_full_fill, resolve_order


def order(status="pending_new", filled_qty="0"):
    return OrderRecord(
        id="order-1", status=status, submitted_at="2026-09-01T14:00:00Z",
        client_order_id="convex-test", filled_qty=filled_qty,
        filled_avg_price="1.25" if filled_qty != "0" else None,
    )


class Gateway:
    def __init__(self, observations):
        self.observations = list(observations)
        self.cancelled = []

    def wait_for_order(self, *_):
        return self.observations.pop(0)

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)


def test_a_filled_status_with_the_requested_quantity_is_a_fill():
    assert is_full_fill(order("filled", "2"), 2)
    assert not is_full_fill(order("filled", "1"), 2)


def test_a_working_order_is_cancelled_and_never_promoted_to_a_fill():
    gateway = Gateway([order("pending_new"), order("canceled")])
    result = resolve_order(gateway, order(), 2, timeout_seconds=1, poll_seconds=1)
    assert gateway.cancelled == ["order-1"]
    assert result.terminal
    assert not result.filled


def test_an_order_still_working_after_cancel_is_explicitly_pending():
    gateway = Gateway([order("pending_new"), order("pending_cancel")])
    result = resolve_order(gateway, order(), 2, timeout_seconds=1, poll_seconds=1)
    assert result.cancel_requested
    assert not result.terminal
    assert not result.filled
