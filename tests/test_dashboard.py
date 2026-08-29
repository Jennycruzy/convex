"""What the page shows, and what it refuses to show.

The dashboard's one job is to be true. Every figure on it is read back out of
the ledger the agent wrote at decision time, so the tests that matter are the
ones proving it cannot display anything the agent did not record — in
particular that an empty ledger produces an empty page and not a demo.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from convex.config import load
from convex.dashboard import read
from convex.dashboard.app import create_app
from convex.dashboard.charts import payoff_svg, waterfall_svg
from convex.ledger import Action, Ledger, Record


@pytest.fixture
def ledger_path(tmp_path, monkeypatch):
    """A config whose ledger points at a temporary file."""
    source = load().path.read_text()
    path = tmp_path / "convex.yaml"
    path.write_text(source.replace("data/ledger/decisions.jsonl", str(tmp_path / "led.jsonl")))
    monkeypatch.setenv("CONVEX_CONFIG", str(path))
    return tmp_path / "led.jsonl"


@pytest.fixture
def client(ledger_path):
    return TestClient(create_app(load(ledger_path.parent / "convex.yaml"))), ledger_path


def write(path, *records):
    ledger = Ledger(path)
    for record in records:
        ledger.append(record)


def refusal(structure="put_bwb", net=-6.0):
    return Record(
        action=Action.CANDIDATE_REJECTED,
        cycle_id="c1",
        structure=structure,
        rationale="Refused: cost exceeded the edge.",
        probability=0.61,
        max_loss=480.0,
        es_contribution=310.0,
        contracts=0,
        reject_reason="net_of_cost",
        extra={
            "waterfall": {
                "gross_edge": 42.0, "half_spread": -18.0, "slippage": -12.0,
                "fees": -4.0, "exit_reserve": -14.0, "net_edge": net,
            }
        },
    )


# ------------------------------------------------------------------ empty state


def test_an_empty_ledger_renders_an_empty_page_and_says_so(client):
    session, _ = client
    page = session.get("/").text
    assert "No decisions recorded yet" in page
    assert "no sample trade and no demo data" in page


def test_the_empty_page_shows_no_figures_at_all(client):
    session, _ = client
    page = session.get("/").text
    assert "realised p&l" not in page
    assert "refusal rate" not in page


def test_a_missing_ledger_file_is_not_an_error(client):
    session, path = client
    assert not path.exists()
    assert session.get("/").status_code == 200
    assert session.get("/api/ledger").json() == []


# ---------------------------------------------------------------------- reading


def test_a_corrupt_ledger_line_raises_rather_than_being_skipped(tmp_path):
    path = tmp_path / "led.jsonl"
    path.write_text('{"seq":1,"action":"stand_down"}\nnot json\n')
    with pytest.raises(ValueError, match="is not valid JSON"):
        read.load(path)


def test_the_summary_counts_refusals_as_decisions(client):
    session, path = client
    write(path, refusal(), refusal("call_bwb"))
    summary = session.get("/api/summary").json()
    assert summary["refusals"] == 2
    assert summary["decisions"] == 2
    assert summary["orders"] == 0


def test_realised_results_accumulate_only_from_settled_positions(client):
    session, path = client
    write(
        path,
        Record(action=Action.POSITION_CLOSED, cycle_id="c1", structure="put_bwb",
               rationale="expired", outcome={"realised_pnl": 120.0}),
        Record(action=Action.POSITION_CLOSED, cycle_id="c2", structure="put_bwb",
               rationale="guard closed a leg", outcome={"order_id": "x", "status": "accepted"}),
    )
    summary = session.get("/api/summary").json()
    assert summary["realised_pnl"] == 120.0
    assert summary["settled_structures"] == 1


def test_the_refusal_rate_is_refusals_over_everything_considered():
    summary = read.Summary(orders=1, refusals=3)
    assert summary.refusal_rate == 0.75
    assert read.Summary().refusal_rate == 0.0


# ----------------------------------------------------------------- the page


def test_a_refusal_appears_on_the_page_with_its_waterfall(client):
    session, path = client
    write(path, refusal())
    page = session.get("/").text
    assert "Gross against net" in page
    assert "cost exceeded the edge" in page
    assert "put_bwb" in page
    assert "refused" in page


def test_the_ledger_endpoint_returns_the_records_unedited(client):
    session, path = client
    write(path, refusal())
    payload = session.get("/api/ledger").json()
    assert payload[0]["reject_reason"] == "net_of_cost"
    assert payload[0]["waterfall"]["gross_edge"] == 42.0


def test_health_check_answers(client):
    session, _ = client
    assert session.get("/healthz").text == "ok"


# --------------------------------------------------------------------- charts


def test_the_waterfall_marks_a_negative_net_differently_from_a_positive_one():
    losing = waterfall_svg(
        {"gross_edge": 42.0, "half_spread": -18.0, "slippage": -12.0,
         "fees": -4.0, "exit_reserve": -14.0, "net_edge": -6.0}
    )
    winning = waterfall_svg(
        {"gross_edge": 42.0, "half_spread": -4.0, "slippage": -3.0,
         "fees": -1.0, "exit_reserve": -2.0, "net_edge": 32.0}
    )
    assert "bar-negative" in losing and "cost exceeded the edge" in losing
    assert "bar-negative" not in winning and "edge survived the cost" in winning


def test_a_waterfall_missing_a_component_raises_rather_than_drawing_a_gap():
    with pytest.raises(ValueError, match="missing"):
        waterfall_svg({"gross_edge": 10.0, "net_edge": 2.0})


def test_the_payoff_diagram_keeps_a_broken_wing_credit_tail_above_zero():
    # +650P / -2x645P / +635P for a 0.20 credit: flat +20 above every strike.
    curve = [(600.0, -480.0), (635.0, -480.0), (645.0, 520.0), (650.0, 20.0), (670.0, 20.0)]
    svg = payoff_svg(curve, breakevens=(636.2, 649.8), spot=650.0)
    assert "pay-fill-up" in svg and "pay-fill-down" in svg
    assert "marker-spot" in svg
    assert svg.count("marker-breakeven") == 2


def test_a_payoff_diagram_needs_more_than_one_point():
    with pytest.raises(ValueError, match="at least two points"):
        payoff_svg([(650.0, 0.0)])
