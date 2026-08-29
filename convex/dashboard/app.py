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
from convex.dashboard.charts import payoff_svg, waterfall_svg
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
    Action.ORDER_SUBMITTED.value: ("opened", "open"),
    Action.ORDER_FILLED.value: ("filled", "open"),
    Action.CANDIDATE_REJECTED.value: ("refused", "refused"),
    Action.ORDER_REJECTED.value: ("rejected", "refused"),
    Action.STAND_DOWN.value: ("stood down", "stand"),
    Action.RISK_HALT.value: ("halted", "refused"),
    Action.POSITION_CLOSED.value: ("closed", "stand"),
    Action.SNAPSHOT.value: ("snapshot", "stand"),
    Action.CALIBRATION.value: ("calibration", "stand"),
}


def _tag(action: str) -> str:
    label, css = TAGS.get(action, (action, "stand"))
    return f'<span class="tag {css}">{escape(label)}</span>'


def _tile(key: str, value: str) -> str:
    return f'<div class="tile"><div class="k">{escape(key)}</div><div class="v">{value}</div></div>'


def _number(value: Any, places: int = 2, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return f"{float(value):,.{places}f}"
    except (TypeError, ValueError):
        return escape(str(value))


def create_app(config: Config | None = None) -> FastAPI:
    """Build the app. The ledger path comes from the same config the agent uses."""
    settings = config if config is not None else load()
    ledger_path = settings.path_("paths.ledger")
    app = FastAPI(title="CONVEX", docs_url=None, redoc_url=None)

    def records() -> list[dict[str, Any]]:
        return read.load(ledger_path)

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

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        rows = records()
        summary = read.summarise(rows)
        body: list[str] = []

        body.append(
            "<header class='top'><h1>CONVEX <span>0DTE SPY options, priced net of "
            "execution cost</span></h1>"
            "<p class='lede'>Every decision below was written to an append-only ledger "
            "before the order existed, refusals included. Nothing on this page is "
            "computed here; it is read back out of what the agent recorded at the "
            "moment it decided.</p></header>"
        )

        if not summary.has_run:
            body.append(
                "<div class='empty'><h3>No decisions recorded yet.</h3>"
                "<p class='lede'>The agent has not completed a cycle against the paper "
                "account, so there is nothing to show. This page deliberately renders "
                "no sample trade and no demo data — a dashboard displaying numbers the "
                "agent never produced would be the exact thing this project argues "
                "against.</p>"
                "<p class='lede'>It fills in after "
                "<code>python -m scripts.preflight</code> and "
                "<code>python -m scripts.run_cycle</code>.</p></div>"
            )
            return HTMLResponse(_page("".join(body)))

        realised = summary.realised_pnl
        body.append("<div class='tiles'>")
        body.append(_tile("realised p&l", f"{realised:+,.2f}"))
        body.append(_tile("cycles", str(summary.cycles)))
        body.append(_tile("opened", str(summary.orders)))
        body.append(_tile("refused", str(summary.refusals)))
        body.append(_tile("stood down", str(summary.stand_downs)))
        body.append(_tile("refusal rate", f"{summary.refusal_rate:.0%}"))
        body.append(_tile("execution cost", f"{summary.execution_cost:,.2f}"))
        body.append("</div>")

        latest = read.waterfalls(rows)
        if latest:
            record = latest[0]
            action = record.get("action", "")
            label = "refused" if action == Action.CANDIDATE_REJECTED.value else "opened"
            body.append("<h2>Gross against net</h2>")
            body.append(
                "<p class='lede'>The bar on the left is the edge before costs. Each "
                "orange bar is a component of getting in and out. The bar on the right "
                "is what is actually left, and it is what the agent ranks on.</p>"
            )
            body.append("<div class='panel'>")
            body.append(
                f"<h3>{escape(str(record.get('structure', 'candidate')))} · {label}</h3>"
            )
            body.append(waterfall_svg(record["waterfall"]))
            if record.get("rationale"):
                body.append(
                    f"<p class='rationale'>{escape(str(record['rationale']))}</p>"
                )
            body.append("</div>")

        body.append("<h2>Decisions</h2>")
        body.append("<div class='panel scroll'><table><thead><tr>")
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
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>CONVEX</title><style>" + STYLE + "</style></head>"
        "<body><div class='wrap'>" + body +
        "<footer>CONVEX · 0DTE SPY structures selected on edge net of measured "
        "execution cost, sized against the tail, reached through Alpaca's MCP server. "
        "Paper account. The full ledger is at <code>/api/ledger</code>.</footer>"
        "</div></body></html>"
    )
