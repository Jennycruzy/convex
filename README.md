# CONVEX

**A cost-aware, tail-budgeted 0DTE SPY options agent that reaches the market entirely through the Model Context Protocol.**

Built for the lablab.ai × Alpaca *AI Trading Agents* hackathon. Paper account only — there is no live-money code path, and the gateway refuses to start if you try.

---

## The thesis

> We tested the obvious trade against ten years of data. Net of costs, it loses money. Here's what actually survives.

Most 0DTE bots are one of two things: an iron condor premium harvester, or a symmetric butterfly. **"0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing"** (SPXW, 09/2016–01/2026, replication package at [vilkovgr/0dte-strategies](https://github.com/vilkovgr/0dte-strategies)) tested both over a decade net of realistic frictions:

| Structure | Gross Sharpe | Net Sharpe |
|---|---:|---:|
| Iron butterfly / condor | 0.77 | **−0.20** |
| Put ratio spread | 1.18 | **0.93** |
| Strangle / straddle | 0.56 | 0.39 |
| Top-three basket | 1.12 | 0.82 |

A long butterfly and an iron butterfly are synthetically equivalent, so the first row is what most of this field will ship. It does not lose because the payoff shape is wrong. It loses because four legs of bid/ask eat the edge.

CONVEX does four things differently:

1. **Trades skew, not variance.** Realized skewness drives 0DTE PnL more than realized variance does. The variance risk premium at 0DTE has a median of ~0.0011% of underlying from 10:00 to expiry — real, and smaller than the cost of harvesting it.
2. **Classifies direction, not magnitude.** Per-structure binary target, L2 logistic, hard mapping — full size or nothing. The paper is explicit that a low-variance parametric model beats higher-capacity ones on short-horizon 0DTE data, so there is no neural net here and that is a deliberate finding, not a shortcut.
3. **Diversifies across structure families.** The basket beat every single structure tested. Nothing concentrates into the highest-probability candidate.
4. **Prices cost and tail risk before entry, not after.** Every candidate is ranked on edge *net* of measured half-spread per leg plus slippage. A structure that is attractive gross and unattractive net is rejected, and the rejection is written to the ledger with its arithmetic.

That last one is the whole project. The gross-to-net gap is where the field dies, and CONVEX makes it visible on every single decision.

---

## What it will not do

- **No symmetric butterflies or iron condors.** Negative net Sharpe. Rejecting them *is* the thesis.
- **No naked shorts, ever.** Every position has a maximum loss computed before the order is built. This is why the primary structure is a put *broken-wing butterfly* rather than a raw put ratio spread: a 1×2 put ratio plus a protective lower wing is the same skew exposure with the risk defined.
- **No stop losses on defined-risk structures.** Stopping out of a capped-loss position converts a bounded loss into a realised loss plus multi-leg slippage. Positions are held to expiry. Exactly three things close one early: the kill switch, the daily loss limit, and the assignment guard.
- **No LLM computing anything.** The language model narrates a brief of figures that were already computed. It cannot produce a price, a Greek, a probability, a size or a risk number, and the agent trades normally when it is switched off.

---

## Translating SPX research onto SPY

Alpaca lists equity and ETF options; index options are not available. The research is SPXW. Every parameter had to be re-derived, and the differences are not cosmetic:

| | SPX (research) | SPY (here) |
|---|---|---|
| Strike increment | $5 | **$1** |
| Settlement | Cash, European | **Physical, American** |
| Assignment risk on shorts | None | **Real** |
| Level | ~6,800 | ~$650 |
| ±2% moneyness band | ±136 pts | **±$13** |

Two consequences run through the code. Every width is expressed as a fraction of spot, never as an absolute number of points. And physical settlement means an open leg at the close can become a hundred shares of stock per contract overnight — so the assignment guard is a hard gate, not a guideline, and it has no counterpart in the paper being implemented.

---

## Architecture

```
Alpaca MCP server (child process, stdio)
  account · clock · calendar · contracts · snapshots · quotes · orders · positions
        │
        ▼
convex/data/mcp.py        one asyncio loop on a background thread, synchronous
convex/data/alpaca.py     JSON becomes typed objects, or raises
        │
        ▼
convex/features.py        10:00 ET snapshot: implied variance, implied skew,
                          slopes, lagged moments, exposure proxies, liquidity
        │
        ▼
convex/classifier.py      per-family L2 logistic → P(net PnL > 0), hard mapping
        │
        ▼
convex/structures/        enumerate candidates per family
convex/costs.py           half-spread per leg + slippage + exit reserve
convex/edge.py            gross edge, net edge, expected shortfall, win rate
        │
        ▼
convex/gates.py           thirteen checks, session-scope and candidate-scope
convex/sizing.py          one function, no override parameter
convex/rationale.py       the written explanation, before the order
convex/ledger.py          append-only JSONL — every decision, including refusals
        │
        ▼
convex/manager.py         hold to expiry; kill switch, loss limit, assignment guard
```

**MCP is the entire data and execution layer.** Not a wrapper around a wrapper — the agent's only route to the market is the server Alpaca publishes, spawned over a pipe, with the toolset restricted to what this project actually trades. Two failure modes get explicit handling because they would otherwise be silent: the order tools answer with a *successful* JSON-RPC result whose body carries an error object, so every payload is inspected before it can be read as a fill; and every call carries a timeout, because a cycle that hangs at 10:00 is a cycle that misses its entry.

---

## The risk checks

Every one must pass, and every verdict — pass or fail — is written to the ledger.

**Session scope**, run once before anything is priced:

1. **Kill switch** — an append-only file, checked every cycle
2. **Market calendar** — from Alpaca's own calendar, never from a local holiday table
3. **Daily loss limit** — 3% of equity, then halt and publish
4. **Buying power** — verified against the account, never assumed
5. **Cost budget** — cumulative execution cost capped at 2% of equity

**Candidate scope**, run on the best candidate of each family:

6. **Max loss computable and within budget** — the agent is structurally incapable of submitting a position whose worst case it cannot compute
7. **Net-of-cost hurdle** — edge must exceed half-spread × legs + slippage. *The most important check in the system.*
8. **Leg-count preference** — two legs beat four at comparable net edge, because each leg is another half-spread
9. **Liquidity** — reject any leg whose relative spread exceeds the measured threshold
10. **Expected shortfall cap** — projected portfolio ES(1%) within 3% of equity
11. **Assignment** — no leg that can settle into shares survives the final thirty minutes
12. **Classifier confidence** — stand down when probabilities cluster at 0.5
13. **Feature staleness** — never trade off a stale chain
14. **Concurrency** — at most four open structures

Standing down is a first-class outcome, logged and published with its reasoning. An agent that knows when not to trade is a stronger result than one that always fires.

---

## Sizing

```
risk_budget = equity × 1%
max_loss    = computed per structure, deterministically
contracts   = floor(risk_budget / max_loss)
             then checked against the portfolio ES(1%) cap
```

One function. No override parameter. Classifier confidence decides *whether* to trade and never *how much* — the research found hard mapping beats confidence-weighted sizing. Size is a function of the tail, not of expected profit: ES(1%) values at 0DTE run roughly 0.58–1.58% of underlying, which makes mean PnL an inadequate summary statistic on its own.

---

## Running it

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env          # then fill in the paper account's keys

.venv/bin/python -m scripts.preflight        # verify every market fact, live
.venv/bin/python -m scripts.calibrate_costs  # measure real SPY spreads per leg
.venv/bin/python -m scripts.run_cycle --dry-run
.venv/bin/python -m scripts.run_cycle        # at 10:00 ET
.venv/bin/python -m scripts.manage           # the guard, through the session
.venv/bin/python -m scripts.manage --settle  # after the close
.venv/bin/python -m scripts.train            # fit on the recorded chains
.venv/bin/python -m scripts.backtest --json  # replay them; --json feeds the dashboard
.venv/bin/python -m scripts.serve            # the dashboard, on :8000
```

Deployment artifacts are in `deploy/` — a systemd unit that keeps the dashboard up
across reboots, bound to loopback and published by nginx which terminates TLS. The
unit runs with `ProtectSystem=strict` and no access to the agent's credentials: the
dashboard only ever reads the ledger.

The dashboard is server-rendered with no build step and no external requests — the
charts are inline SVG, because a deployed demo that renders a blank panel when a CDN
is slow scores as a demo that does not work. Before the agent has run it shows an
empty page saying so; there is no sample trade and no demo mode.

`--dry-run` is not a simulation. Every read goes to the real API; only the write is withheld, so it exercises the same chain, the same account and the same checks as a live cycle.

Tests:

```bash
.venv/bin/python -m pytest
```

The suite starts the real Alpaca MCP server as a subprocess and asserts against the
tool surface it actually exposes. Nothing in this repository mocks a chain, stubs a
quote, or returns a canned price.

The integration suite talks to the live paper account and is run separately:

```bash
.venv/bin/python -m pytest tests/integration -s
```

It skips cleanly without credentials. With them it measures the real per-leg spread,
checks the strike increment and multiplier against the chain rather than the
configuration, reads Alpaca's own calendar for the competition window, runs a full
decision cycle with only the write withheld, and asserts that a candidate really is
refused because its execution cost consumed its edge — a check that has never been
observed firing has not been demonstrated. Use `-s` to see the measured figures.

---

## Honest limitations

- **The research is SPX; this trades SPY.** Every parameter carried across is a hypothesis. Values in `config/convex.yaml` are marked `MEASURED` or `HYPOTHESIS`, and the hypotheses are not to be relied on in a live decision until `scripts/calibrate_costs.py` has replaced them.
- **The exposure features are proxies.** They are flow/exposure estimates built from traded volume, open interest and leg Greeks — not a dealer-inventory reconstruction. Calling them GEX would overstate what they are.
- **The classifier may not have enough history.** Training runs on chains the agent *recorded*, never on simulated ones, because an expired contract's book cannot be fetched back — a session that was not recorded can never be labelled honestly. Early on there will be too few sessions to clear the burn-in; `scripts/train.py` says so and fits nothing rather than presenting a model fitted on thirty rows. The agent then runs a documented volatility-regime rule, and every ledger record states which of the two made the call.
- **No Sharpe is reported below twenty observations.** A handful of trades with
  similar results produces a ratio in the hundreds; that is a small denominator, not
  an edge, and printing it would be the most misleading number this project could
  publish. Over a four-session window this means no Sharpe is reported at all, which
  is the correct outcome rather than a gap.
- **Four trading sessions is not a track record.** P&L across this window is substantially variance. The reproducible parts of this project are the cost discipline, the risk checks and the receipts.

---

## Licence

MIT.
