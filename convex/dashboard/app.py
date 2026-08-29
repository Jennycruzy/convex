"""The deployed dashboard.

One page, server-rendered, no build step and no external requests. It reads the
decision ledger and shows what the agent did: the features it saw at 10:00, the
probability it assigned each structure family, every candidate it priced, the
gross-to-net waterfall that decided the outcome, the payoff of anything it
opened, the rationale it wrote before acting, and every refusal.

The refusals are not a footnote here. They are given the same weight as the
fills, because an agent that declines a trade whose edge its own execution
would have eaten is demonstrating the thesis, and a reader who has only ever
seen gross backtests has probably never watched that happen.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from convex.config import Config, load
from convex.dashboard import read
from convex.dashboard import ui
from convex.dashboard.charts import payoff_svg, sensitivity_svg, waterfall_svg
from convex.ledger import Action

STYLE = """
:root {
  --ink: #10151c; --muted: #5d6b7e; --line: #d9e0e8; --panel: #ffffff;
  --ground: #f4f6f9; --accent: #1f6feb; --good: #10796b; --bad: #b4342b;
  --cost: #d98324;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--ground); color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 32px 20px 72px; }
header.top { border-bottom: 2px solid var(--ink); padding-bottom: 18px; margin-bottom: 28px; }
h1 { margin: 0; font-size: 30px; letter-spacing: -0.02em; }
h1 span { color: var(--muted); font-weight: 400; font-size: 17px; letter-spacing: 0; }
h2 { font-size: 19px; margin: 34px 0 12px; letter-spacing: -0.01em; }
h3 { font-size: 15px; margin: 0 0 8px; }
p.lede { color: var(--muted); margin: 8px 0 0; max-width: 76ch; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }
.tile .k { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
.tile .v { font-size: 25px; font-variant-numeric: tabular-nums; margin-top: 4px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 18px 20px; margin-bottom: 16px; }
.chart { width: 100%; height: auto; display: block; }
.axis { stroke: #9aa7b6; stroke-width: 1; }
.bar-total { fill: var(--good); } .bar-negative { fill: var(--bad); } .bar-cost { fill: var(--cost); }
.bar-value { font-size: 11px; fill: var(--ink); text-anchor: middle; font-variant-numeric: tabular-nums; }
.bar-label { font-size: 11px; fill: var(--muted); text-anchor: middle; }
.axis-label { font-size: 10px; fill: var(--muted); }
.chart-note { font-size: 12px; fill: var(--muted); }
.pay-line { fill: none; stroke: var(--ink); stroke-width: 2; }
.pay-fill-up { fill: rgba(16,121,107,.16); } .pay-fill-down { fill: rgba(180,52,43,.14); }
.marker-breakeven { stroke: var(--accent); stroke-width: 1; stroke-dasharray: 3 3; }
.marker-spot { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 4; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th { text-align: left; color: var(--muted); font-weight: 600; font-size: 11.5px;
  text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid var(--line); padding: 8px 10px; }
td { padding: 9px 10px; border-bottom: 1px solid #eef1f5; vertical-align: top; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.tag { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 10px;
  border: 1px solid var(--line); white-space: nowrap; }
.tag.open { background: #e6f4f1; border-color: #a9d5cd; color: var(--good); }
.tag.refused { background: #fdeceb; border-color: #f0bdb8; color: var(--bad); }
.tag.stand { background: #eef2f7; color: var(--muted); }
.rationale { font-size: 13.5px; color: #2a323d; }
.scroll { overflow-x: auto; }
.empty { border: 1px dashed #b9c4d1; border-radius: 8px; padding: 26px 22px; background: #fbfcfd; }
.empty code { background: #eef1f5; padding: 1px 6px; border-radius: 4px; font-size: 13px; }
footer { margin-top: 44px; padding-top: 16px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 12.5px; }
@media (prefers-color-scheme: dark) {
  :root { --ink: #e8edf3; --muted: #93a1b2; --line: #2b3542; --panel: #161c24;
          --ground: #0e1319; --accent: #6ea8ff; --good: #4bbfa8; --bad: #e8776c; --cost: #e2a552; }
  td { border-bottom-color: #222b36; }
  .tag.open { background: #12281f; border-color: #2d5a45; }
  .tag.refused { background: #2a1614; border-color: #5c2f28; }
  .tag.stand { background: #1a212a; }
  .empty { background: #131920; border-color: #33404f; }
  .empty code { background: #1e2732; }
}
"""

TAGS = {
    Action.ORDER_SUBMITTED.value: ("opened", "opened"),
    Action.ORDER_FILLED.value: ("filled", "opened"),
    Action.CANDIDATE_REJECTED.value: ("refused", "refused"),
    Action.ORDER_REJECTED.value: ("rejected", "refused"),
    Action.STAND_DOWN.value: ("stood down", "stood"),
    Action.RISK_HALT.value: ("halted", "halt"),
    Action.POSITION_CLOSED.value: ("closed", "stood"),
    Action.SNAPSHOT.value: ("snapshot", "stood"),
    Action.CALIBRATION.value: ("calibration", "stood"),
}


def _tag(action: str) -> str:
    label, css = TAGS.get(action, (action, "stood"))
    return f'<span class="badge {css}">{escape(label)}</span>'


def _tile(key: str, value: str, note: str = "", count: float | None = None,
          places: int = 0, prefix: str = "", signed: bool = False) -> str:
    """One headline figure.

    ``count`` opts the figure into the count-up: the rendered text is already
    the final value, so a reader with JavaScript off, or one who asked for less
    motion, sees the number rather than a zero that never animates.
    """
    attrs = ""
    if count is not None:
        attrs = (
            f" data-count='{count}' data-places='{places}'"
            f" data-prefix='{escape(prefix)}'" + (" data-signed" if signed else "")
        )
    tail = f"<div class='tile-note'>{escape(note)}</div>" if note else ""
    return (
        f"<div class='tile'><div class='tile-key'>{escape(key)}</div>"
        f"<div class='tile-value num'{attrs}>{value}</div>{tail}</div>"
    )


def _number(value: Any, places: int = 2, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return f"{float(value):,.{places}f}"
    except (TypeError, ValueError):
        return escape(str(value))


def _sharpe(value) -> str:
    """A Sharpe, or a dash when the sample was too small to support one."""
    return "—" if value is None else f"{float(value):.2f}"


def _backtest_panel(report: dict) -> str:
    """The gross-against-net table, which is the argument in one picture."""
    body = [
        _section("evidence", "REPLAY, GROSS AGAINST NET"),
        "<p>The same comparison the research makes, run on the chains "
        "this agent recorded. Every row is measured twice: once before execution "
        "cost and once after. The gap between the two columns is where most 0DTE "
        "strategies quietly stop working.</p>",
    ]
    sessions = int(report.get("sessions", 0))
    if sessions < 30:
        body.append(
            f"<div class='empty'><p>Replayed over {sessions} session(s). "
            "That is far too few for a Sharpe ratio to carry meaning, so none is shown "
            "below twenty observations — a handful of similar trades produces a ratio "
            "in the hundreds, which is a small denominator rather than an edge.</p></div>"
        )

    body.append("<div class='panel scroll-x reveal'><table><thead><tr>")
    for column in ("", "trades", "gross SR", "net SR", "gross $", "net $", "cost $", "survives"):
        body.append(f"<th>{column}</th>")
    body.append("</tr></thead><tbody>")

    def row(label: str, arm: dict, emphasis: bool = False) -> None:
        survives = arm.get("net_sharpe")
        verdict = (
            "<span class='tag stand'>—</span>" if survives is None
            else ("<span class='tag open'>yes</span>" if survives > 0
                  else "<span class='tag refused'>no</span>")
        )
        name = f"<strong>{escape(label)}</strong>" if emphasis else escape(label)
        body.append(
            f"<tr><td>{name}</td>"
            f"<td class='num'>{arm.get('trades', 0)}</td>"
            f"<td class='num'>{_sharpe(arm.get('gross_sharpe'))}</td>"
            f"<td class='num'>{_sharpe(arm.get('net_sharpe'))}</td>"
            f"<td class='num'>{_number(arm.get('gross_total'))}</td>"
            f"<td class='num'>{_number(arm.get('net_total'))}</td>"
            f"<td class='num'>{_number(arm.get('cost_total'))}</td>"
            f"<td>{verdict}</td></tr>"
        )

    for family, arms in sorted(report.get("per_family", {}).items()):
        for name, arm in arms.items():
            row(f"{family} · {name}", arm)
    for name, arm in sorted(report.get("basket", {}).items()):
        row(arm.get("label", name), arm, emphasis=True)

    body.append("</tbody></table></div>")
    return "".join(body)


def _cycle_panel(cycle) -> str:
    """What the agent thought of each family this cycle, and what it did."""
    verdicts = cycle.verdicts
    if not verdicts:
        return ""
    body = [
        _section("last pass", "WHAT IT MADE OF EACH FAMILY"),
        "<p>One row per structure family: the probability it was given, "
        "which of the two decided it, and what happened. Standing down is an outcome "
        "here, not a missing row.</p>",
        "<div class='panel scroll-x reveal'><table><thead><tr>",
    ]
    for column in ("structure", "p", "decided by", "outcome", "lots", "reason"):
        body.append(f"<th>{column}</th>")
    body.append("</tr></thead><tbody>")
    for record in verdicts:
        body.append(
            f"<tr><td>{escape(str(record.get('structure') or '—'))}</td>"
            f"<td class='num'>{_number(record.get('probability'), 3)}</td>"
            f"<td>{escape(str(record.get('probability_source') or '—'))}</td>"
            f"<td>{_tag(str(record.get('action', '')))}</td>"
            f"<td class='num'>{_number(record.get('contracts'), 0)}</td>"
            f"<td>{escape(str(record.get('reject_reason') or ''))}</td></tr>"
        )
    body.append("</tbody></table></div>")
    return "".join(body)


def create_app(config: Config | None = None) -> FastAPI:
    """Build the app. The ledger path comes from the same config the agent uses."""
    settings = config if config is not None else load()
    ledger_path = settings.path_("paths.ledger")
    app = FastAPI(title="CONVEX", docs_url=None, redoc_url=None)

    backtest_path = settings.path_("paths.backtest_report")
    sensitivity_path = backtest_path.parent / "sensitivity.json"

    def records() -> list[dict[str, Any]]:
        return read.load(ledger_path)

    def sensitivity() -> dict:
        """The spread sweep, if one has been run. Absent is not an error."""
        if not sensitivity_path.is_file():
            return {}
        return json.loads(sensitivity_path.read_text())

    def backtest_report() -> dict:
        """The replay, if one has been run. Absent is not an error."""
        if not backtest_path.is_file():
            return {}
        return json.loads(backtest_path.read_text())

    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz() -> str:
        return "ok"

    @app.get("/api/ledger")
    def api_ledger() -> JSONResponse:
        """The whole ledger, unedited. The page is a view; this is the source."""
        return JSONResponse(records())

    @app.get("/api/summary")
    def api_summary() -> JSONResponse:
        return JSONResponse(read.summarise(records()).__dict__)

    @app.get("/api/backtest")
    def api_backtest() -> JSONResponse:
        return JSONResponse(backtest_report())

    @app.get("/api/sensitivity")
    def api_sensitivity() -> JSONResponse:
        return JSONResponse(sensitivity())

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        rows = records()
        summary = read.summarise(rows)
        body: list[str] = []

        body.append(_masthead(summary, summary.has_run))
        body.append(_hero(summary, sensitivity()))
        body.append(
            _section(
                "receipts",
                "EVERY DECISION, INCLUDING EVERY REFUSAL",
                "Written to an append-only ledger before the order existed. Nothing "
                "on this page is computed here — it is read back out of what the "
                "agent recorded at the moment it decided.",
            )
        )

        if not summary.has_run:
            body.append(
                "<div class='panel reveal'><div class='panel-body'>"
                "<h3>NO DECISIONS RECORDED YET</h3>"
                "<p>The agent has not completed a cycle against the paper "
                "account, so there is nothing to show. This page deliberately renders "
                "no sample trade and no demo data — a dashboard displaying numbers the "
                "agent never produced would be the exact thing this project argues "
                "against.</p>"
                "<p class='faint'>It fills in after "
                "<code>python -m scripts.preflight</code> and "
                "<code>python -m scripts.run_cycle</code>.</p></div></div>"
            )
            return HTMLResponse(_page("".join(body)))

        realised = summary.realised_pnl
        tone = "good" if realised > 0 else ("bad" if realised < 0 else "")
        body.append("<div class='grid tiles stagger'>")
        body.append(
            _tile(
                "realised p&l",
                f"<span class='{tone}'>{realised:+,.2f}</span>",
                f"{summary.settled_structures} settled",
            )
        )
        body.append(_tile("cycles", str(summary.cycles), count=summary.cycles))
        body.append(_tile("opened", str(summary.orders), count=summary.orders))
        body.append(
            _tile("refused", str(summary.refusals), "cost ate the edge",
                  count=summary.refusals)
        )
        body.append(
            _tile("stood down", str(summary.stand_downs), "a first-class outcome",
                  count=summary.stand_downs)
        )
        body.append(
            _tile("refusal rate", f"{summary.refusal_rate:.0%}",
                  "low is not obviously good")
        )
        body.append(
            _tile("execution cost", f"{summary.execution_cost:,.2f}", "paid to trade",
                  count=summary.execution_cost, places=2)
        )
        body.append("</div>")

        latest = read.waterfalls(rows)
        if latest:
            record = latest[0]
            action = record.get("action", "")
            label = "refused" if action == Action.CANDIDATE_REJECTED.value else "opened"
            body.append(_section("mechanism", "GROSS AGAINST NET"))
            body.append(
                "<p>The bar on the left is the edge before costs. Each "
                "orange bar is a component of getting in and out. The bar on the right "
                "is what is actually left, and it is what the agent ranks on.</p>"
            )
            body.append("<div class='panel reveal'>")
            body.append(
                "<div class='panel-head'><h3>"
                f"{escape(str(record.get('structure', 'candidate')))}</h3>"
                f"{_tag(action)}</div>"
            )
            body.append("<div class='panel-body'>")
            body.append(waterfall_svg(record["waterfall"]))
            if record.get("rationale"):
                body.append(
                    f"<p class='note' style='margin-top:16px'>{escape(str(record['rationale']))}</p>"
                )
            body.append("</div></div>")

        opened = [
            record for record in reversed(rows)
            if record.get("action") == Action.ORDER_SUBMITTED.value and record.get("legs")
        ]
        if opened:
            record = opened[0]
            curve, strikes = read.payoff_from_record(record)
            spot = (record.get("features") or {}).get("spot")
            body.append(
                _section(
                    "position",
                    "WHAT WAS OPENED",
                    "Value at expiry across underlying prices, recomputed from the "
                    "receipt rather than stored. The flat tail above every strike is "
                    "a broken-wing butterfly entered for a credit: up there it keeps "
                    "the credit and risks nothing, which is exactly what it buys over "
                    "a ratio spread with an open downside.",
                )
            )
            body.append("<div class='panel reveal'>")
            body.append(
                f"<h3>{escape(str(record.get('structure', 'structure')))} · "
                f"{_number(record.get('contracts'), 0)} lots · strikes "
                f"{escape(', '.join(f'{strike:g}' for strike in strikes))}</h3>"
            )
            body.append("<div class='panel-body'>")
            body.append(payoff_svg(curve, breakevens=strikes, spot=spot))
            body.append(
                f"<p class='note' style='margin-top:16px'>Worst case "
                f"{_number(record.get('max_loss'))} · ES(1%) "
                f"{_number(record.get('es_contribution'))} · entered at "
                f"{_number(record.get('net_price'))} net.</p>"
            )
            body.append("</div></div>")

        cycles = read.cycles(rows)
        if cycles:
            body.append(_cycle_panel(cycles[0]))

        replay = backtest_report()
        if replay.get("per_family") or replay.get("basket"):
            body.append(_backtest_panel(replay))

        body.append("<h2>Decisions</h2>")
        body.append("<div class='panel scroll-x reveal'><table><thead><tr>")
        for column in (
            "time", "outcome", "structure", "p", "net edge",
            "max loss", "ES(1%)", "lots", "reason",
        ):
            body.append(f"<th>{column}</th>")
        body.append("</tr></thead><tbody>")

        shown = [
            record for record in reversed(rows)
            if record.get("action") in {action.value for action in read.DECISION_ACTIONS}
        ][:60]
        for record in shown:
            waterfall = record.get("waterfall") or {}
            body.append("<tr>")
            body.append(f"<td>{escape(read.format_stamp(record.get('ts')))}</td>")
            body.append(f"<td>{_tag(str(record.get('action', '')))}</td>")
            body.append(f"<td>{escape(str(record.get('structure') or '—'))}</td>")
            body.append(f"<td class='num'>{_number(record.get('probability'), 3)}</td>")
            body.append(f"<td class='num'>{_number(waterfall.get('net_edge'))}</td>")
            body.append(f"<td class='num'>{_number(record.get('max_loss'))}</td>")
            body.append(f"<td class='num'>{_number(record.get('es_contribution'))}</td>")
            body.append(f"<td class='num'>{_number(record.get('contracts'), 0)}</td>")
            body.append(f"<td>{escape(str(record.get('reject_reason') or ''))}</td>")
            body.append("</tr>")
            if record.get("rationale"):
                body.append(
                    f"<tr><td></td><td colspan='8' class='rationale'>"
                    f"{escape(str(record['rationale']))}</td></tr>"
                )
        body.append("</tbody></table></div>")

        return HTMLResponse(_page("".join(body)))

    return app


def _page(body: str) -> str:
    """The whole document. One response, no second request for anything."""
    return (
        "<!doctype html><html lang='en' data-theme='dark'><head>"
        "<meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta name='color-scheme' content='dark light'>"
        "<title>CONVEX — 0DTE SPY, priced net of cost</title>"
        "<meta name='description' content='A cost-aware, tail-budgeted 0DTE SPY "
        "options agent. Every decision, including every refusal, with the "
        "arithmetic that produced it.'>"
        "<script>" + ui.HEAD_SCRIPT + "</script>"
        "<style>" + ui.stylesheet() + "</style>"
        "</head><body><div class='wrap'>" + body +
        "<footer class='foot'>"
        "<div>CONVEX · 0DTE SPY structures ranked on edge <strong>net</strong> of "
        "measured execution cost, sized against the tail, reached entirely through "
        "Alpaca's MCP server. Paper account.</div>"
        "<div>The page is a view. The source is "
        "<a href='/api/ledger'>/api/ledger</a>.</div>"
        "</footer></div>"
        "<script>" + ui.SCRIPT + "</script>"
        "</body></html>"
    )


def _masthead(summary, live: bool) -> str:
    """The status bar.

    A terminal tells you the state of the world before it tells you anything
    else, so this is the first row on the page and it stays there while you
    scroll: what this is, whether the ledger is live, what it has done, and the
    exchange clock — which is the only thing on the page that moves without a
    reload, and is there because a terminal that does not tick looks frozen.
    """
    led = "led live" if live else "led off"
    state = "LEDGER LIVE" if live else "NO CYCLE YET"
    return (
        "<div class='statusbar'>"
        "<div class='mark'>CON<b>V</b>EX</div>"
        f"<div><span class='{led}'></span><span class='stat-val'>{state}</span></div>"
        "<div><span class='stat-key'>UND</span>"
        "<span class='stat-val'>SPY 0DTE</span></div>"
        "<div><span class='stat-key'>ACCT</span>"
        "<span class='stat-val'>PAPER</span></div>"
        f"<div><span class='stat-key'>CYC</span>"
        f"<span class='stat-val'>{summary.cycles}</span></div>"
        f"<div><span class='stat-key'>REF</span>"
        f"<span class='stat-val down'>{summary.refusals}</span></div>"
        f"<div><span class='stat-key'>OPN</span>"
        f"<span class='stat-val up'>{summary.orders}</span></div>"
        "<div class='spacer'></div>"
        "<div><span class='stat-key'>CLK</span>"
        "<span class='stat-val' data-clock>--:--:--</span></div>"
        "<button class='theme-toggle' data-theme-toggle type='button'>LIGHT</button>"
        "</div>"
    )


def _section(label: str, heading: str, lede: str = "") -> str:
    """A section head that spans the page rather than stacking down one edge.

    The heading takes the left field and whatever explains it takes the right,
    so the header row is as wide as the data underneath it and the eye is never
    walked past an empty half to reach the next thing.
    """
    if not lede:
        return (
            f"<section class='reveal'><div class='rule'>{escape(label)}</div>"
            f"<h2>{heading}</h2></section>"
        )
    return (
        f"<section class='reveal'><div class='rule'>{escape(label)}</div>"
        "<div style='display:grid;gap:18px 40px;align-items:start;"
        "grid-template-columns:repeat(auto-fit,minmax(280px,1fr))'>"
        f"<h2>{heading}</h2><p style='margin:0'>{lede}</p>"
        "</div></section>"
    )


def _hero(summary, sensitivity: dict) -> str:
    """The claim, with the evidence beside it rather than under it.

    Two fields across the page: the argument on the left at a readable measure,
    the decade of research it rests on tabulated on the right. A reader who
    only ever looks at the right-hand column still leaves knowing the finding,
    because the finding is a pair of numbers in the last two rows.
    """
    # The published figures this project is built on. Ten years, SPXW, net of
    # realistic frictions. Quoted, not recomputed.
    research = (
        ("iron butterfly / condor", "0.77", "-0.20", True),
        ("put ratio spread", "1.18", "0.93", False),
        ("strangle / straddle", "0.56", "0.39", False),
        ("top-three basket", "1.12", "0.82", False),
    )
    rows = "".join(
        f"<tr><td>{escape(name)}</td>"
        f"<td class='num'>{gross}</td>"
        f"<td class='num {'down' if dead else 'up'}'>{net}</td></tr>"
        for name, gross, net, dead in research
    )

    parts = [
        "<section class='reveal' style='margin-top:4px'>",
        "<div class='rule'>the finding</div>",
        "<div class='split scan'>",
        "<div class='prose'>",
        "<h1><span class='type shift'>GROSS, IT WORKS.</span><br>"
        "<span class='down'>NET, IT DOESN'T.</span></h1>",
        "<p class='lede' style='margin-top:16px'>The obvious 0DTE structures were "
        "tested over ten years. Before costs they look like a strategy. After "
        "costs the best-known one has a <strong class='down'>negative</strong> "
        "Sharpe. The payoff shape was never the problem — four legs of bid-ask "
        "were.</p>",
        "<p>CONVEX reruns that test on SPY, over sessions rebuilt from the option "
        "tape, and ranks every candidate on edge <em>after</em> the spread it "
        "would pay to get in. Attractive gross and unattractive net is a "
        "refusal, and the refusal is published with the arithmetic that "
        "produced it.</p>",
        "</div>",
        "<div>",
        "<div class='rule' style='margin-bottom:10px'>ten years, SPXW, net of frictions</div>",
        "<table><thead><tr><th>structure</th>"
        "<th class='num'>gross SR</th><th class='num'>net SR</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        "<p class='faint' style='font-size:var(--t-micro);margin:10px 0 0;"
        "letter-spacing:0.08em'>THE FIRST ROW IS WHAT MOST 0DTE BOTS SHIP.</p>",
        "</div>",
        "</div></section>",
    ]

    points = (sensitivity or {}).get("points") or []
    if points:
        crossing = _crossing(points)
        parts.append(
            "<section class='reveal'>"
            "<div class='rule'>where the edge dies</div>"
            "<div class='panel scan'>"
            "<div class='panel-head'>"
            "<h3>net sharpe against spread paid per leg</h3>"
            + (
                f"<div><span class='stat-key'>BREAK-EVEN</span> "
                f"<span class='stat-val cost'>{crossing}</span></div>"
                if crossing
                else ""
            )
            + "</div><div class='panel-body'>"
            + sensitivity_svg(points)
            + "<div class='scrub-readout' data-scrub-readout>"
            "<div><div class='k'>spread / leg</div>"
            "<div class='v' data-f='spread'>--</div></div>"
            "<div><div class='k'>gross sharpe</div>"
            "<div class='v' data-f='gross'>--</div></div>"
            "<div><div class='k'>net sharpe</div>"
            "<div class='v' data-f='net'>--</div></div>"
            "<div><div class='k'>sessions traded</div>"
            "<div class='v' data-f='trades'>--</div></div>"
            "</div>"
            + "<p class='note' style='margin-top:16px'><strong>Read this before "
            "quoting it.</strong> These sessions were rebuilt from trade prints, "
            "not recorded from the live book, and the spread is modelled rather "
            "than measured — the book for those sessions is gone. Every point is "
            "a full replay; nothing between them is drawn. Which side of the "
            "crossing SPY actually trades on is a measurement, taken at the open, "
            "not an argument had here.</p>"
            "</div></div></section>"
        )
    return "".join(parts)


def _crossing(points: list) -> str:
    """The spread at which the classified basket's net Sharpe changes sign.

    Reported as the bracket the sweep actually resolves it to. Quoting a single
    interpolated number would claim a precision the eight measured points do
    not carry.
    """
    last_positive = None
    first_negative = None
    for point in points:
        net = (point.get("classified") or {}).get("net_sharpe")
        if net is None:
            continue
        if net > 0:
            last_positive = point["relative_spread"]
        elif first_negative is None and last_positive is not None:
            first_negative = point["relative_spread"]
    if last_positive is None:
        return ""
    if first_negative is None:
        return f"beyond {last_positive:.1%}"
    return f"{last_positive:.1%}–{first_negative:.1%}"
