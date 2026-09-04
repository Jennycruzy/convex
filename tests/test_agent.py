"""Decision-cycle execution, including the fresh-quote retry path."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, time
from types import SimpleNamespace

import numpy as np

import convex.agent as agent_module
from convex.agent import Agent, CycleResult, PricedCandidate
from convex.config import load
from convex.costs import CostModel
from convex.data.alpaca import OrderRecord
from convex.edge import evaluate
from convex.gates import GateContext, GateReport, GateResult
from convex.ledger import Action, Ledger
from convex.rationale import Rationale
from convex.scenarios import ScenarioSet
from convex.sizing import SizeDecision
from convex.structures.base import Candidate, Family


def test_canceled_zero_fill_reprices_from_fresh_quotes_and_records_the_fill(
    put_bwb_legs, tmp_path, monkeypatch
):
    config = load()
    family = Family.PUT_BWB
    candidate = Candidate(
        family=family,
        legs=tuple(put_bwb_legs),
        description="test put broken-wing butterfly",
    )
    scenarios = ScenarioSet(
        log_returns=np.linspace(-0.01, 0.01, 50),
        source_days=tuple(date(2026, 1, 1) for _ in range(50)),
        entry_time=time(10, 0),
        exit_time=time(16, 0),
        volatility_scale=1.0,
        built_at=datetime(2026, 8, 28, 14, 0, tzinfo=UTC),
    )
    estimate = evaluate(
        candidate.legs,
        scenarios,
        CostModel.from_config(config),
        650.0,
        1,
        0.01,
    )
    best = PricedCandidate(candidate, estimate)
    size = SizeDecision(
        contracts=1,
        risk_budget=1_000.0,
        max_loss_per_contract=estimate.profile.max_loss,
        es_per_contract=estimate.expected_shortfall,
        es_headroom=3_000.0,
        buying_power_per_contract=estimate.profile.max_loss,
        binding_constraint="profile_contract_cap",
    )
    report = GateReport(
        [GateResult("test_gate", "candidate", True, "test gate passed")]
    )
    now = datetime(2026, 8, 28, 14, 0, tzinfo=UTC)
    context = GateContext(
        config=config,
        now_exchange=now,
        session_close=datetime(2026, 8, 28, 20, 0, tzinfo=UTC),
        is_trading_day=True,
        market_open=True,
        equity=100_000.0,
        last_equity=100_000.0,
        buying_power=200_000.0,
        spot=650.0,
        open_structures=0,
        es_in_use=0.0,
        cumulative_fees=0.0,
        kill_switch_path=tmp_path / "KILL",
        probability=0.75,
    )
    account = SimpleNamespace(
        equity=100_000.0,
        last_equity=100_000.0,
        options_buying_power=200_000.0,
    )
    fresh_quotes = {
        leg.contract.symbol: replace(
            leg.entry.quote,
            bid=leg.entry.quote.bid + 0.02,
            ask=leg.entry.quote.ask + 0.02,
            timestamp=now,
        )
        for leg in candidate.legs
    }

    class Gateway:
        def __init__(self):
            self.submissions = []
            self.waited_for = []
            self.quote_calls = 0
            self.account_calls = 0

        def account(self):
            self.account_calls += 1
            return account

        def clock(self):
            return now, True

        def spot(self, symbol):
            assert symbol == "SPY"
            return 650.0, None

        def option_quotes(self, symbols):
            self.quote_calls += 1
            assert list(symbols) == [leg.contract.symbol for leg in candidate.legs]
            return fresh_quotes

        def submit_structure(self, legs, contracts, limit_price, client_order_id):
            self.submissions.append((legs, contracts, limit_price, client_order_id))
            order_id = f"entry-{len(self.submissions)}"
            return OrderRecord(
                id=order_id,
                status="new",
                submitted_at="2026-08-28T14:00:00Z",
                client_order_id=client_order_id,
                filled_qty="0",
                filled_avg_price=None,
            )

        def wait_for_order(self, order_id, *_):
            self.waited_for.append(order_id)
            status = "canceled" if order_id == "entry-1" else "filled"
            return OrderRecord(
                id=order_id,
                status=status,
                submitted_at="2026-08-28T14:00:00Z",
                client_order_id="test-client",
                filled_qty="0" if status == "canceled" else "1",
                filled_avg_price=None if status == "canceled" else "1.25",
            )

        def cancel_order(self, order_id):
            raise AssertionError(
                f"terminal canceled order should not be canceled again: {order_id}"
            )

    gateway = Gateway()
    passed_session = GateReport(
        [GateResult("test_session", "session", True, "session passed")]
    )
    passed_candidate = GateReport(
        [GateResult("test_candidate", "candidate", True, "candidate passed")]
    )
    monkeypatch.setattr(agent_module, "run_session_gates", lambda context: passed_session)
    monkeypatch.setattr(
        agent_module, "run_candidate_gates", lambda *args: passed_candidate
    )
    monkeypatch.setattr(agent_module, "size_position", lambda *args: size)
    monkeypatch.setattr(
        agent_module.rationale_layer,
        "narrate",
        lambda brief, fallback: Rationale(fallback, "deterministic", brief),
    )

    agent = Agent(
        gateway=gateway,
        config=config,
        ledger=Ledger(tmp_path / "retry.jsonl"),
        scenarios=scenarios,
        reprice_ticks=(1, 2),
    )
    result = CycleResult("cycle-1", True, "no structure opened")

    agent._open(
        "cycle-1",
        family,
        best,
        size,
        report,
        0.75,
        "test profile",
        result,
        context,
    )

    assert result.orders == [{"family": "put_bwb", "order_id": "entry-2"}]
    assert len(gateway.submissions) == 2
    assert gateway.submissions[1][3].endswith("-r1")
    assert (
        gateway.submissions[1][0][0].entry.quote
        == fresh_quotes[candidate.legs[0].contract.symbol]
    )
    assert gateway.account_calls == 1
    assert gateway.quote_calls == 1
    assert gateway.waited_for == ["entry-1", "entry-2"]

    records = list(agent.ledger.read())
    assert [record["action"] for record in records] == [
        Action.ORDER_SUBMITTED.value,
        Action.ORDER_REJECTED.value,
        Action.ORDER_SUBMITTED.value,
        Action.ORDER_FILLED.value,
    ]
    assert records[1]["outcome"]["status"] == "canceled"
    assert records[1]["outcome"]["filled_qty"] == "0"
    assert records[1]["reprice_enabled"] is True
    assert records[2]["reprice_attempt"] == 1
    assert records[3]["reprice_attempt"] == 1
    assert records[3]["outcome"]["status"] == "filled"


def test_retry_client_ids_preserve_the_attempt_suffix_at_the_broker_limit():
    cycle_id = "20260903T140000-" + "a" * 48

    initial = agent_module._entry_client_order_id(cycle_id, Family.DEBIT_VERTICAL)
    retry_one = agent_module._entry_client_order_id(cycle_id, Family.DEBIT_VERTICAL, 1)
    retry_two = agent_module._entry_client_order_id(cycle_id, Family.DEBIT_VERTICAL, 2)

    assert all(len(client_id) <= 48 for client_id in (initial, retry_one, retry_two))
    assert len({initial, retry_one, retry_two}) == 3
    assert retry_one.endswith("-r1")
    assert retry_two.endswith("-r2")
