"""What the page shows, and what it refuses to show.

The dashboard's one job is to be true. Every figure on it is read back out of
the ledger the agent wrote at decision time, so the tests that matter are the
ones proving it cannot display anything the agent did not record, in
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
    assert "no decisions recorded yet" in page.lower()
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


def test_a_correction_removes_an_invalidated_loss_from_the_dashboard(client):
    session, path = client
    write(
        path,
        Record(action=Action.POSITION_CLOSED, cycle_id="c1", structure="call_bwb",
               rationale="incorrect settlement", outcome={"realised_pnl": -840.0}),
        Record(action=Action.CORRECTION, cycle_id="audit-c1", structure="call_bwb",
               rationale="The canceled entry never filled; this settlement is invalid.",
               extra={"invalidates": [1]}),
    )
    summary = session.get("/api/summary").json()
    assert summary["realised_pnl"] == 0.0
    assert summary["settled_structures"] == 0
    assert read.realised_curve(read.load(path)) == ([], [])


def test_the_refusal_rate_is_refusals_over_everything_considered():
    summary = read.Summary(orders=1, refusals=3)
    assert summary.refusal_rate == 0.75
    assert read.Summary().refusal_rate == 0.0


# ----------------------------------------------------------------- the page


def test_a_refusal_appears_on_the_page_with_its_waterfall(client):
    session, path = client
    write(path, refusal())
    page = session.get("/").text
    assert "gross against net" in page.lower()
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


# --------------------------------------------------------------- payoff wiring


def opened_bwb():
    return Record(
        action=Action.ORDER_FILLED,
        cycle_id="c9",
        structure="put_bwb",
        rationale="Entering 2 lots.",
        probability=0.64,
        contracts=2,
        cost_breakdown={"total": 8.40, "half_spread": 6.0, "slippage": 2.4},
        net_price=-0.20,
        max_loss=960.0,
        es_contribution=620.0,
        legs=[
            {"symbol": "SPY260831P00650000", "right": "put", "strike": 650.0, "ratio": 1},
            {"symbol": "SPY260831P00645000", "right": "put", "strike": 645.0, "ratio": -2},
            {"symbol": "SPY260831P00635000", "right": "put", "strike": 635.0, "ratio": 1},
        ],
        outcome={"status": "filled", "filled_qty": "2", "order_id": "filled-bwb"},
        extra={"waterfall": {"gross_edge": 40.0, "half_spread": -6.0, "slippage": -4.0,
                             "fees": -1.0, "exit_reserve": -3.0, "net_edge": 26.0}},
    )


def test_the_payoff_is_rebuilt_from_the_receipt_not_refetched():
    curve, strikes = read.payoff_from_record(opened_bwb().__dict__ | {"legs": opened_bwb().legs})
    assert strikes == (635.0, 645.0, 650.0)
    # Two lots of a 0.20 credit: +40 above every strike, bounded at -960 below.
    assert max(value for _, value in curve) > 0
    assert min(value for _, value in curve) == pytest.approx(-960.0)
    assert [value for price, value in curve if price >= 660][0] == pytest.approx(40.0)


def test_a_record_with_no_legs_cannot_be_drawn():
    with pytest.raises(ValueError, match="no legs"):
        read.payoff_from_record({"net_price": 1.0})


def test_a_leg_missing_its_strike_raises_rather_than_drawing_a_wrong_shape():
    with pytest.raises(ValueError, match="missing 'strike'"):
        read.payoff_from_record({"legs": [{"right": "put", "ratio": 1}], "net_price": 1.0})


def test_an_opened_structure_gets_a_payoff_panel_on_the_page(client):
    session, path = client
    write(path, opened_bwb())
    page = session.get("/").text
    assert "what was opened" in page.lower()
    assert "635, 645, 650" in page
    assert "pay-line" in page


def test_a_refusal_is_featured_ahead_of_a_winning_fill(client):
    session, path = client
    write(path, refusal("straddle"), opened_bwb())
    ordered = read.waterfalls(read.load(path))
    assert ordered[0]["structure"] == "straddle"
    page = session.get("/").text
    assert "cost exceeded the edge" in page


# ------------------------------------------------------------------- the replay


def backtest_payload():
    return {
        "sessions": 4,
        "per_family": {
            "put_bwb": {
                "every session": {
                    "trades": 4, "gross_sharpe": None, "net_sharpe": None,
                    "gross_total": 210.0, "net_total": -40.0, "cost_total": 250.0,
                },
            },
        },
        "basket": {
            "every session": {
                "label": "basket, every session", "trades": 4,
                "gross_sharpe": 0.77, "net_sharpe": -0.20,
                "gross_total": 210.0, "net_total": -40.0, "cost_total": 250.0,
            },
        },
    }


@pytest.fixture
def with_backtest(ledger_path, tmp_path):
    """A config whose replay report exists on disk."""
    path = tmp_path / "convex.yaml"
    path.write_text(
        path.read_text().replace("data/backtest.json", str(tmp_path / "bt.json"))
    )
    (tmp_path / "bt.json").write_text(json.dumps(backtest_payload()))
    return TestClient(create_app(load(path))), ledger_path


def test_a_missing_replay_is_not_an_error(client):
    session, path = client
    write(path, refusal())
    assert session.get("/api/backtest").json() == {}
    assert "Replay, gross against net" not in session.get("/").text


def test_the_replay_shows_gross_and_net_side_by_side(with_backtest):
    session, path = with_backtest
    write(path, refusal())
    page = session.get("/").text
    assert "replay, gross against net" in page.lower()
    assert "0.77" in page and "-0.20" in page


def test_an_arm_that_does_not_survive_its_costs_is_marked_as_such(with_backtest):
    session, path = with_backtest
    write(path, refusal())
    page = session.get("/").text
    # The basket has a negative net Sharpe, so its verdict must read "no".
    assert "tag refused'>no<" in page


def test_a_replay_over_too_few_sessions_says_so_rather_than_implying_a_result(with_backtest):
    session, path = with_backtest
    write(path, refusal())
    page = session.get("/").text
    assert "too few for a Sharpe ratio to carry meaning" in page


def test_a_sharpe_that_could_not_be_computed_renders_as_a_mark_not_a_zero(with_backtest):
    """An absent figure is marked absent. A zero would be a claim."""
    from convex.dashboard.app import _sharpe

    assert _sharpe(None) == "·"
    assert _sharpe(0.77) == "0.77"


# -------------------------------------------------------------- the last cycle


def test_the_last_cycle_names_which_of_the_two_decided_each_family(client):
    session, path = client
    write(
        path,
        Record(action=Action.CANDIDATE_REJECTED, cycle_id="c7", structure="straddle",
               rationale="Refused.", probability=0.4, reject_reason="classifier_confidence",
               extra={"probability_source": "regime rule (high_variance)"}),
        Record(action=Action.ORDER_SUBMITTED, cycle_id="c7", structure="put_bwb",
               rationale="Entering.", probability=0.64, contracts=2,
               cost_breakdown={"total": 8.40},
               extra={"probability_source": "classifier"}),
    )
    page = session.get("/").text
    assert "what it made of each family" in page.lower()
    assert "regime rule (high_variance)" in page
    assert "classifier" in page
    assert "classifier_confidence" in page


# ------------------------------------------------------- the sensitivity chart


def _sweep_points():
    """A sweep whose net Sharpe changes sign between two and three per cent."""
    return [
        {"relative_spread": 0.01, "classified": {"net_sharpe": 0.59, "gross_sharpe": 1.28, "trades": 118}},
        {"relative_spread": 0.02, "classified": {"net_sharpe": 0.58, "gross_sharpe": 1.60, "trades": 94}},
        {"relative_spread": 0.03, "classified": {"net_sharpe": -0.10, "gross_sharpe": 1.45, "trades": 80}},
        {"relative_spread": 0.05, "classified": {"net_sharpe": -0.68, "gross_sharpe": 1.35, "trades": 79}},
    ]


def test_the_crossing_is_reported_as_the_bracket_the_sweep_resolves():
    """Quoting one interpolated number would claim precision eight points lack."""
    from convex.dashboard.app import _crossing

    assert _crossing(_sweep_points()) == "2.0% to 3.0%"


def test_a_sweep_that_never_turns_negative_says_so_rather_than_inventing_a_crossing():
    from convex.dashboard.app import _crossing

    points = [
        {"relative_spread": 0.01, "classified": {"net_sharpe": 0.8, "gross_sharpe": 1.2, "trades": 10}},
        {"relative_spread": 0.02, "classified": {"net_sharpe": 0.4, "gross_sharpe": 1.1, "trades": 10}},
    ]
    assert _crossing(points).startswith("beyond")


def test_the_sensitivity_chart_marks_where_the_edge_dies():
    from convex.dashboard.charts import sensitivity_svg

    svg = sensitivity_svg(_sweep_points())
    assert svg.startswith("<svg")
    assert "edge dies here" in svg
    # Every measured point is drawn, and each carries its own figures. Counted
    # by class so the scrubber's own marker is not mistaken for a data point.
    assert svg.count("class='pt'") == 4
    assert "net Sharpe +0.59" in svg


def test_the_chart_refuses_to_draw_a_curve_from_one_point():
    from convex.dashboard.charts import sensitivity_svg

    assert "not produced enough points" in sensitivity_svg(_sweep_points()[:1])


def test_the_page_carries_its_own_styles_and_makes_no_external_request(client):
    page = client[0].get("/").text
    assert "<style>" in page and "cubic-bezier" in page
    for scheme in ("http://", "https://", "//cdn"):
        assert f'src="{scheme}' not in page and f"src='{scheme}" not in page
    # Motion is opt-out, and the opt-out is honoured rather than declared.
    assert "prefers-reduced-motion" in page


def test_the_theme_choice_is_offered_and_applied_before_paint(client):
    page = client[0].get("/").text
    assert "data-theme-toggle" in page
    assert "convex-theme" in page
    assert page.index("convex-theme") < page.index("<style>")


def test_every_animated_class_is_actually_observed():
    """The bug this exists to catch: an effect defined but never triggered.

    Each of these classes does nothing until the observer adds "in" to it. A
    class styled in the stylesheet but missing from the observer's selector is
    an animation that silently never runs, which is exactly what happened to
    the travelling rule and the scanline.
    """
    from convex.dashboard import ui

    triggered = {
        line.split(".in")[0].strip().lstrip(".")
        for line in ui.BASE.splitlines()
        if ".in" in line and line.strip().startswith(".")
    }
    watched = ui.SCRIPT[ui.SCRIPT.index("querySelectorAll("):]
    watched = watched[: watched.index(")")]
    missing = sorted(
        name for name in triggered if name and f".{name}" not in watched
    )
    assert not missing, f"styled for animation but never observed: {missing}"


def test_the_page_is_set_at_a_readable_size():
    """13px monospace is a diff view, not a page anybody reads."""
    from convex.dashboard import ui

    base = ui.TOKENS.split("--t-base:")[1].split("px")[0].strip()
    assert float(base) >= 15


def test_the_two_column_field_gives_both_columns_the_same_room():
    from convex.dashboard import ui

    block = ui.BASE[ui.BASE.index(".split {"):ui.BASE.index(".split > *")]
    assert "repeat(2, minmax(0, 1fr))" in block


def test_every_colour_the_charts_ask_for_is_a_colour_the_palette_defines():
    """The bug this exists to catch: charts painting black on black.

    An undefined custom property in an SVG fill resolves to the initial value,
    which is black, and on this ground that is invisible rather than wrong-
    looking. When the palette was renamed the charts kept asking for the old
    names and their axis labels and dots simply disappeared.
    """
    import re

    from convex.dashboard import charts, ui

    wanted = set(re.findall(r"var\((--[a-z-]+)\)", charts.__file__ and
                            open(charts.__file__).read()))
    defined = set(re.findall(r"^\s+(--[a-z-]+):", ui.TOKENS, re.MULTILINE))
    missing = sorted(wanted - defined)
    assert not missing, f"charts ask for undefined colours: {missing}"


# ------------------------------------------------------------------- the log


def test_the_log_reads_the_way_a_log_is_written(client):
    """Oldest first, so a new session appends to the bottom.

    The ordering is the whole reason this is a log rather than a table: the
    reader's eye stays where it was and tomorrow's cycle lands underneath.
    """
    session, path = client
    write(
        path,
        Record(action=Action.CANDIDATE_REJECTED, cycle_id="c1", structure="put_bwb",
               reject_reason="net_of_cost", rationale="Cost ate it."),
        Record(action=Action.ORDER_SUBMITTED, cycle_id="c2", structure="call_bwb",
               rationale="Opened.", contracts=2,
               cost_breakdown={"total": 8.40}),
    )
    page = session.get("/").text
    assert "decision log" in page.lower()
    # Scoped to the log itself: both structures also appear in the panels above
    # it, so a page-wide index would be measuring the wrong thing.
    log = page[page.index("<div class='log' data-log>"):]
    log = log[: log.index("append-only")]
    assert log.index("put_bwb") < log.index("call_bwb")


def test_each_day_is_marked_once_as_it_turns(client):
    session, path = client
    write(
        path,
        Record(action=Action.STAND_DOWN, cycle_id="c1", rationale="Nothing cleared."),
        Record(action=Action.STAND_DOWN, cycle_id="c2", rationale="Nor here."),
    )
    page = session.get("/").text
    # Two entries on one day produce one day header, not two.
    assert page.count("class='log-day'") == 1
    assert "2 decisions" in page


def test_the_log_scrolls_in_its_own_frame_rather_than_growing_the_page():
    """A week of sessions has to be a scroll, not a redesign."""
    from convex.dashboard import ui

    block = ui.BASE[ui.BASE.index(".log {"):ui.BASE.index(".log::-webkit-scrollbar ")]
    assert "overflow-y: auto" in block and "max-height" in block
    assert "[data-log]" in ui.SCRIPT


def test_a_refusal_carries_its_reason_into_the_log(client):
    session, path = client
    write(
        path,
        Record(action=Action.CANDIDATE_REJECTED, cycle_id="c1", structure="straddle",
               reject_reason="liquidity", rationale="Spread too wide to cross."),
    )
    page = session.get("/").text
    assert "Spread too wide to cross." in page
    assert "refused" in page


def test_the_receipts_heading_sits_on_the_log_it_names(client):
    """It used to introduce the whole lower page while the log was far below.

    A heading that names a thing has to be adjacent to it, or it reads as a
    promise about something else.
    """
    session, path = client
    write(path, Record(action=Action.STAND_DOWN, cycle_id="c1", rationale="Nothing."))
    page = session.get("/").text
    heading = page.index("EVERY DECISION")
    log = page.index("<div class='log' data-log>")
    assert heading < log
    # Nothing but the panel head may come between them.
    assert log - heading < 1200


def test_the_log_is_introduced_once_and_not_twice(client):
    """A heading left behind by the table this replaced sat above the section's
    own heading, so the log was announced twice under two different names."""
    session, path = client
    write(path, Record(action=Action.STAND_DOWN, cycle_id="c1", rationale="Nothing."))
    page = session.get("/").text
    assert "<h2>Decisions</h2>" not in page
    assert page.count("EVERY DECISION, INCLUDING EVERY REFUSAL") == 1


# ----------------------------------------------------- the realised equity curve


def closed(pnl, cost=None, cycle="c1"):
    outcome = {"realised_pnl": pnl}
    if cost is not None:
        outcome["execution_cost"] = cost
    return Record(
        action=Action.POSITION_CLOSED,
        cycle_id=cycle,
        structure="put_bwb",
        rationale="expired",
        outcome=outcome,
    )


def test_the_realised_curve_is_empty_until_a_position_actually_closes(client):
    session, path = client
    write(path, refusal(), Record(
        action=Action.STAND_DOWN, cycle_id="c1", rationale="calibration refused"
    ))
    gross, net = read.realised_curve(read.load(path))
    assert gross == [] and net == []

    page = session.get("/").text
    assert "REALISED, THIS ACCOUNT" in page
    assert "placed no orders yet" in page


def test_the_empty_realised_panel_draws_no_curve_at_all(client):
    """Not a flat line at zero, which would read as having traded to breakeven."""
    session, path = client
    write(path, refusal())
    page = session.get("/").text
    head, _, _ = page.partition("REPLAY, GROSS AGAINST NET")
    _, _, realised = head.partition("REALISED, THIS ACCOUNT")
    assert "<svg" not in realised


def test_the_realised_curve_accumulates_closes_in_ledger_order(client):
    session, path = client
    write(path, closed(120.0), closed(-45.0), closed(30.0))
    gross, net = read.realised_curve(read.load(path))
    assert net == [120.0, 75.0, 105.0]
    # No cost recorded on any close, so the two curves coincide rather than
    # showing a band that was never measured.
    assert gross == net


def test_a_close_carrying_its_cost_separates_the_two_curves(client):
    session, path = client
    write(path, closed(100.0, cost=20.0), closed(-10.0, cost=15.0))
    gross, net = read.realised_curve(read.load(path))
    assert net == [100.0, 90.0]
    assert gross == [120.0, 125.0]


def test_a_close_without_its_number_adds_no_point_rather_than_a_zero(client):
    session, path = client
    write(
        path,
        closed(50.0),
        Record(action=Action.POSITION_CLOSED, cycle_id="c2", structure="put_bwb",
               rationale="guard closed a leg", outcome={"status": "accepted"}),
    )
    _, net = read.realised_curve(read.load(path))
    assert net == [50.0]


def test_the_realised_panel_draws_once_there_are_two_closes(client):
    session, path = client
    write(path, closed(120.0), closed(-45.0))
    page = session.get("/").text
    head, _, _ = page.partition("REPLAY, GROSS AGAINST NET")
    _, _, realised = head.partition("REALISED, THIS ACCOUNT")
    assert "<svg" in realised
    assert "+75.00 net" in realised
    assert "2 structure(s) settled" in realised


def test_the_break_even_bracket_stops_at_the_first_crossing():
    """A later point coming back above zero is not the break-even.

    The sweep's widest point turns positive again on a handful of trades, which
    is the classifier's own gate selecting down to a sample that stopped meaning
    anything. Walking past the first crossing let that point become the left
    edge and printed the bracket backwards.
    """
    from convex.dashboard.app import _crossing

    points = [
        {"relative_spread": 0.005, "classified": {"net_sharpe": 0.95}},
        {"relative_spread": 0.010, "classified": {"net_sharpe": 0.64}},
        {"relative_spread": 0.015, "classified": {"net_sharpe": -0.42}},
        {"relative_spread": 0.050, "classified": {"net_sharpe": -1.75}},
        {"relative_spread": 0.070, "classified": {"net_sharpe": 0.10}},
    ]

    assert _crossing(points) == "1.0% to 1.5%"


def test_a_curve_that_never_crosses_says_so():
    from convex.dashboard.app import _crossing

    points = [
        {"relative_spread": 0.005, "classified": {"net_sharpe": 0.95}},
        {"relative_spread": 0.010, "classified": {"net_sharpe": 0.64}},
    ]

    assert _crossing(points) == "beyond 1.0%"


def test_the_waterfall_leads_with_a_refusal_cost_actually_killed():
    """The panel argues the project's case, so it must lead with a case.

    A candidate refused on a wide leg while its edge survived cost captions
    itself "edge survived the cost", which reads as cost refusing something
    cost approved.
    """
    from convex.dashboard import read

    survived = {
        "ts": "2026-09-01T14:13:10+00:00",
        "action": "candidate_rejected",
        "structure": "debit_vertical",
        "reject_reason": "liquidity",
        "waterfall": {"gross_edge": 67.09, "net_edge": 61.99},
    }
    killed = {
        "ts": "2026-08-31T17:22:10+00:00",
        "action": "candidate_rejected",
        "structure": "put_bwb",
        "reject_reason": "net_of_cost",
        "waterfall": {"gross_edge": 7.22, "net_edge": -101.06},
    }

    ordered = read.waterfalls([survived, killed])

    assert ordered[0]["reject_reason"] == "net_of_cost"
