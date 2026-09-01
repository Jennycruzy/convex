"""A whole decision cycle, with the write withheld.

Everything reads from the live account: the clock, the calendar, the chain, the
Greeks, the buying power. Only the order is not sent. That is the same wrapper
the live script uses, so a passing run here means the 10:00 cycle will reach
the same conclusions from the same data.
"""

from __future__ import annotations

import pytest

from convex.agent import Agent
from convex.classifier import load_models
from convex.errors import ConvexError
from convex.ledger import Action, Ledger
from convex.scenarios import build as build_scenarios
from scripts.run_cycle import DryRunGateway, family_results
from tests.integration.conftest import needs_account


@needs_account
def test_a_full_cycle_reaches_a_decision_and_records_every_verdict(
    gateway, config, tmp_path
):
    ledger = Ledger(tmp_path / "cycle.jsonl")
    scenarios = build_scenarios(gateway, config)
    print(f"  scenarios: {scenarios.describe()}")

    now, _ = gateway.clock()
    sessions = gateway.sessions(now.date(), now.date())
    if not sessions:
        pytest.skip(f"no session on {now.date()}; a cycle cannot run")

    models, _ = load_models(config.path_("paths.models"), config)
    print(f"  {len(models)} fitted model(s); the rest use the documented rule")

    agent = Agent(
        gateway=DryRunGateway(gateway),
        config=config,
        ledger=ledger,
        scenarios=scenarios,
        models=models,
    )

    try:
        result = agent.run_cycle(
            prior_returns=scenarios.log_returns.tolist(),
            variance_history=scenarios.annualised_variance().tolist(),
            family_pnl=family_results(ledger),
        )
    except ConvexError as error:
        # A dry run refuses the write by raising, which is the correct
        # behaviour once a candidate has cleared everything.
        assert "dry run" in str(error), f"the cycle failed for a real reason: {error}"
        print(f"  reached the order: {error}")
        result = None

    records = list(ledger.read())
    print(f"  {len(records)} ledger record(s) written")
    assert records, "a cycle that reached no conclusion wrote no receipt"

    for record in records:
        assert record["rationale"], f"a {record['action']} record carries no rationale"
        assert record["cycle_id"]

    if result is not None:
        print(f"  {result.reason}")
        for rejection in result.rejections:
            print(f"    refused {rejection['family']}: {rejection['reason']}")


@needs_account
def test_the_cycle_archives_the_chain_it_decided_from(gateway, config):
    """Without the recording, this session can never be labelled."""
    from convex import archive

    now, _ = gateway.clock()
    recorded = archive.sessions(config.path_("paths.chain_archive"))
    if now.date() not in recorded:
        pytest.skip("no cycle has run today yet, so there is no recording to check")

    snapshot = archive.read(archive.path_for(config.path_("paths.chain_archive"), now.date()))
    print(f"  {len(snapshot.entries)} contracts recorded at {snapshot.taken_at}")
    assert snapshot.spot > 0.0
    if not any(entry.greeks is not None for entry in snapshot.entries):
        pytest.xfail(
            "Alpaca supplied no expiry-day Greeks; the raw snapshot preserves that "
            "fact and the cycle uses its deterministic implied-volatility solver"
        )
