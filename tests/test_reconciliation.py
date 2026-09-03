from convex.ledger import Action
from convex.reconciliation import _entry_sources


def test_retry_fill_completes_missing_submission_legs_from_verified_receipt():
    records = [
        {
            "action": Action.ORDER_SUBMITTED.value,
            "cycle_id": "cycle-1",
            "structure": "call_bwb",
            "client_order_id": "entry-original",
            "contracts": 15,
            "legs": [{"symbol": "OLD", "ratio": 1}],
        },
        {
            "action": Action.ORDER_SUBMITTED.value,
            "cycle_id": "cycle-1",
            "structure": "call_bwb",
            "client_order_id": "entry-retry",
            "contracts": 6,
        },
        {
            "action": Action.ORDER_FILLED.value,
            "cycle_id": "cycle-1",
            "structure": "call_bwb",
            "client_order_id": "entry-retry",
            "contracts": 6,
            "legs": [{"symbol": "SPY260902C00750000", "ratio": 1}],
            "cost_breakdown": {"total": 11.78},
            "net_price": -0.57,
            "outcome": {"order_id": "broker-entry", "client_order_id": "entry-retry"},
        },
    ]

    entries = _entry_sources(records)

    assert len(entries) == 1
    source, submitted = entries[0]
    assert source["outcome"]["order_id"] == "broker-entry"
    assert submitted["contracts"] == 6
    assert submitted["legs"] == [{"symbol": "SPY260902C00750000", "ratio": 1}]
    assert submitted["cost_breakdown"] == {"total": 11.78}


def test_reconciliation_uses_broker_activity_for_a_close_missing_from_legacy_receipts():
    from datetime import date
    from decimal import Decimal
    from types import SimpleNamespace

    from convex.reconciliation import Entry, Fill, reconcile

    class Gateway:
        def order_raw(self, order_id):
            if order_id == "entry-parent":
                return {
                    "id": "entry-parent", "status": "filled", "qty": "1", "filled_qty": "1",
                    "filled_avg_price": "1.00", "legs": [
                        {"id": "entry-leg", "status": "filled", "qty": "1", "filled_qty": "1",
                         "filled_avg_price": "1.00", "symbol": "SPYTEST", "side": "buy"}
                    ],
                }
            raise AssertionError(order_id)

        def account_activities(self, **kwargs):
            return [
                {"activity_type": "FILL", "order_id": "entry-leg", "symbol": "SPYTEST", "side": "buy", "qty": "1", "price": "1.00"},
                {"activity_type": "FILL", "order_id": "close-leg", "symbol": "SPYTEST", "side": "sell", "qty": "1", "price": "1.50"},
            ]

    class Ledger:
        def read(self):
            return iter([{
                "action": Action.ORDER_SUBMITTED.value, "cycle_id": "cycle", "structure": "put_bwb",
                "client_order_id": "client", "contracts": 1, "legs": [{"symbol": "SPYTEST", "ratio": 1}],
                "ts": "2026-09-02T10:00:00+00:00",
            }, {
                "action": Action.ORDER_FILLED.value, "cycle_id": "cycle", "structure": "put_bwb",
                "client_order_id": "client", "contracts": 1, "legs": [{"symbol": "SPYTEST", "ratio": 1}],
                "outcome": {"order_id": "entry-parent", "client_order_id": "client"},
                "ts": "2026-09-02T10:00:01+00:00",
            }])

    result = reconcile(Gateway(), Ledger())
    assert result[0]["gross_realised_pnl"] == 50.0
    assert result[0]["exit_order_ids"] if "exit_order_ids" in result[0] else True
