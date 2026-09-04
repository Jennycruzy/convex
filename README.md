# CONVEX

**A cost-aware, tail-budgeted 0DTE (zero days to expiration) SPY options agent that reaches the market entirely through the Model Context Protocol.**


> ## Start here — hackathon submission
>
> **[Open the live demo →](https://convex.isobars.xyz)**<br>
> **[Watch the demo video →](https://x.com/jennyoliver57/status/2095880230338453784)**<br>
> **[Read the one-page build brief →](docs/WRITEUP.md)**<br>
> **[Download the presentation →](https://convex.isobars.xyz/download/convex-deck)**
>
> The build brief covers the required **AI logic, risk gates, and Alpaca MCP infrastructure**. The dashboard shows the same system’s decisions, economics, gates, broker outcomes, and append-only receipts.

Built for the lablab.ai × Alpaca *AI Trading Agents* hackathon. Paper account only. There is no live-money code path, and the gateway refuses to start if you try.

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

The first row is representative of the four-leg neutral structures most of this field will ship. It does not lose because the payoff shape is wrong. It loses because four legs of bid/ask eat the edge.

CONVEX does four things differently:

1. **Trades skew, not variance.** Realized skewness drives 0DTE PnL more than realized variance does. The variance risk premium at 0DTE has a median of ~0.0011% of underlying from 10:00 to expiry: real, and smaller than the cost of harvesting it.
2. **Classifies direction, not magnitude.** Per-structure binary target, L2 logistic, hard mapping, full size or nothing. The paper is explicit that a low-variance parametric model beats higher-capacity ones on short-horizon 0DTE data, so there is no neural net here and that is a deliberate finding, not a shortcut.
3. **Requires chronological evidence.** No family may create new risk until it beats cash on an untouched post-selection segment after costs. The 2 September audit failed both BWB families, so they remain disabled. The only installed entry profile is a one-lot gap-continuation debit vertical: it enables a direction-matched candidate in memory only after a qualifying opening gap and VWAP check, and it still has to clear every execution and risk gate. It can—and often should—stand down.
4. **Prices cost and tail risk before entry, not after.** Every candidate is ranked on edge *net* of measured half-spread per leg plus slippage. A structure that is attractive gross and unattractive net is rejected, and the rejection is written to the ledger with its arithmetic.

That last one is the whole project. The gross-to-net gap is where the field dies, and CONVEX makes it visible on every single decision.

---

## What it will not do

- **No BWB or iron-condor comeback.** The BWB families failed the untouched audit and stay disabled. The gap-continuation profile is an explicitly bounded paper observation, not a claim that the old strategy was repaired.
- **No naked shorts, ever.** Every position has a maximum loss computed before the order is built. The currently installed profile uses a one-lot debit vertical, while the disabled BWB and straddle families remain unavailable to the live entry path.
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

Two consequences run through the code. Every width is expressed as a fraction of spot, never as an absolute number of points. And physical settlement means an open leg at the close can become a hundred shares of stock per contract overnight, so the assignment guard is a hard gate, not a guideline, and it has no counterpart in the paper being implemented.

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
convex/gates.py           sixteen checks, session-scope and candidate-scope
convex/sizing.py          one function, no override parameter
convex/rationale.py       the written explanation, before the order
convex/ledger.py          append-only JSONL, every decision including refusals
        │
        ▼
convex/manager.py         hold to expiry; kill switch, loss limit, assignment guard
```

**MCP is the entire data and execution layer.** Not a wrapper around a wrapper: the agent's only route to the market is the server Alpaca publishes, spawned over a pipe, with the toolset restricted to what this project actually trades. Two failure modes get explicit handling because they would otherwise be silent: the order tools answer with a *successful* JSON-RPC result whose body carries an error object, so every payload is inspected before it can be read as a fill; and every call carries a timeout, because a cycle that hangs at 10:00 is a cycle that misses its entry.

---

## The risk checks

Every blocking check must pass, and every verdict—including the non-blocking leg-count preference—is written to the ledger.

**Session scope**, run once before anything is priced:

1. **Kill switch**: an append-only file, checked every cycle
2. **Calibration**: no position is opened while any cost or liquidity input is still an unmeasured guess. The spread threshold is measured at 09:55 ET and a fresh successful receipt is required at the exact 10:00 entry. Fee assumptions remain conservative bounds until Alpaca's activity provides the actual fee receipt
3. **Market calendar**: from Alpaca's own calendar, never from a local holiday table
4. **Daily loss limit**: 3% of equity, then halt and publish
5. **Buying power**: verified against the account, never assumed
6. **Cost budget**: cumulative execution cost capped at 2% of equity

**Candidate scope**, run on the best candidate of each family:

7. **Max loss computable and within budget**: the agent is structurally incapable of submitting a position whose worst case it cannot compute
8. **Net-of-cost hurdle**: edge must exceed half-spread × legs + slippage. *The most important check in the system.*
9. **Positive net-edge bound**: the candidate's 95% lower bound on scenario mean net P&L must remain positive
10. **Leg-count preference**: two legs beat four at comparable net edge; this is audit-visible and non-blocking because each leg is another half-spread
11. **Liquidity**: reject any leg whose relative spread exceeds the lower of the observed book threshold and the validated 1% admission cap
12. **Expected shortfall cap**: projected portfolio ES(1%) within 3% of equity
13. **Assignment**: no leg that can settle into shares survives the final thirty minutes
14. **Classifier confidence**: stand down when probabilities cluster at 0.5
15. **Feature staleness**: never trade off a stale chain
16. **Concurrency**: at most one attributable structure, so an assignment guard can close it atomically

Standing down is a first-class outcome, logged and published with its reasoning. An agent that knows when not to trade is a stronger result than one that always fires.

---

## Sizing

```
risk_budget = equity × 1%
max_loss    = computed per structure, deterministically
contracts   = floor(risk_budget / max_loss)
             then checked against the portfolio ES(1%) cap
```

One function. No override parameter. Classifier confidence decides *whether* to trade and never *how much*. The research found hard mapping beats confidence-weighted sizing. Size is a function of the tail, not of expected profit: ES(1%) values at 0DTE run roughly 0.58–1.58% of underlying, which makes mean PnL an inadequate summary statistic on its own.

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
.venv/bin/python -m scripts.profile_backtest --days 800  # read-only gap/VWAP walk-forward replay
.venv/bin/python -m scripts.serve            # the dashboard, on :8000
```

---

## The problem this solves

0DTE options create a particularly persuasive failure mode: a candidate can show a large profit at a midpoint quote while being untradeable after the bid/ask, fees, slippage, and the cost of closing short legs. A payoff diagram can be correct and the trade can still be economically wrong. Physical settlement adds a second failure mode for SPY: a short option that survives the close can become stock in the account.

There is also a research failure mode. If a strategy is selected on the same days used to celebrate it, or if a historical replay silently assumes a book that no longer exists, a positive total is not evidence. Finally, an LLM that is allowed to invent a price, probability, or size turns a reproducible decision into an unreviewable prompt outcome.

## What CONVEX actually solves

CONVEX is a decision and evidence layer for that problem:

- It converts MCP JSON into typed market facts and fails closed when a quote, Greek, calendar, or broker response is missing or malformed.
- It prices every candidate net of per-leg execution cost, reserves exit cost, computes maximum loss and expected shortfall, and refuses positive-gross candidates whose costs consume the edge.
- It treats physical settlement and assignment as first-class risk. Defined risk is computed before submission; the manager watches the close and can close attributable positions before shares arrive.
- It makes chronology explicit. Recorded chains are the basis for training; reconstructed research is point-in-time; threshold selection happens strictly before the applied session; and an untouched holdout is required for promotion.
- It treats standing down, canceled zero-fill entries, risk halts, and corrections as evidence. The append-only ledger records the arithmetic instead of hiding the refusal.
- It keeps the language model outside the numerical control plane. The model can explain already-computed facts, but it cannot supply an order price, Greek, probability, position size, or risk number.

This is not a claim that the current profile is profitable. The current evidence says the opposite: the previously traded families lost after costs, and the replacement profile has not yet cleared its walk-forward promotion bar. That is an intentional safety result, not an omitted result.

## Submission snapshot

As of 3 September 2026, the paper account is flat with no open orders. The installed policy is:

- BWB, straddle, and the other generic structure families are disabled in `config/convex.yaml`.
- The scheduled entry profile is `gap_continuation_vertical`. It enables only a one-lot debit vertical in memory, and only after a deterministic opening-gap/VWAP signal. The production structure list remains empty, so the legacy generic runner cannot create that risk accidentally.
- The signal is evaluated from completed stock bars, and a qualifying candidate still has to pass the normal session and candidate gates. No signal is a valid stand-down.
- A terminal canceled entry with zero contracts filled may use the configured fresh-quote ladder `[1, 2]`. Each rung re-reads account, clock, spot, and option quotes, re-runs the gates, re-sizes, and records its own broker outcome. It never turns a partial fill into a position.

The last broker-reconciled fills were negative:

| Session | Structure | Contracts | Net P&L |
|---|---|---:|---:|
| 1 Sep | put BWB | 14 | −$565.30 |
| 1 Sep | straddle | 7 | −$281.32 |
| 2 Sep | call BWB | 6 | −$180.00 |

Total verified realized P&L: **−$1,026.62**. A separate correction receipt retires a canceled 0/10 entry rather than counting it as a loss. No result here is presented as a positive-return guarantee.

## The replacement profile: evidence, not a promise

The small earlier trend-conditioned sample was **+$454.34 over 26 trades**, but its one-sided 95% lower bound was **−$46.36**. A positive point total with a negative lower bound is not a promotion signal, so the profile was isolated and tested rather than declared a winner.

The profile-specific evaluator, `scripts/profile_backtest.py`, was run over 800 requested days using reconstructed option prints and point-in-time stock bars. Its audited run produced 551 reconstructed sessions, 548 usable observations, a 488-session training segment, and a 60-session untouched holdout:

| Test | Net P&L | Trades | 95% lower bound | Decision |
|---|---:|---:|---:|---|
| Configured baseline, training | −$2,407.44 | — | −$132.47 | Do not promote |
| Configured baseline, untouched holdout | +$122.73 | 3 | −$413.38 | Do not promote |
| Best raw training threshold | +$304.92 | 26 | −$93.18 | Not a promotion decision |
| Walk-forward selection path | −$2,284.71 | 50 | −$125.42 | Do not promote |

No threshold cleared the promotion bar. The default bar requires enough training trades, a positive training lower bound, at least ten untouched holdout trades, positive holdout net P&L, and a positive holdout lower bound. The walk-forward path selects only from sessions strictly earlier than the session it applies to. These results are why the current profile remains one lot, paper-only, and bounded.

## Walk-forward evaluator

The evaluator is research-only. It does not submit orders, write the live ledger, or mutate `config/convex.yaml`. It can optionally write a JSON research report to a path explicitly supplied by the caller.

It performs the following checks:

1. Rebuild each historical option session from prints and use the same debit-vertical candidate construction and cost model as the live path.
2. Build the gap/VWAP signal from stock bars available for that historical session. The observation builder uses the minimum threshold in the tested grid, so lower custom thresholds cannot be silently omitted.
3. Build the scenario distribution only from sessions strictly before the decision date.
4. Measure each threshold on an earlier training segment and an untouched holdout. The baseline is read from `strategy.gap_continuation`, not duplicated in source.
5. Run an expanding walk-forward selection path and report warmup, fallback, threshold choices, trade count, net P&L, lower bound, drawdown, and expected shortfall.

Historical option books are not available. The command therefore models a uniform per-leg spread from each historical option print. It can test the economics of a candidate under that assumption; it cannot prove historical depth, displayed size, quote age, or the exact fill a live order would have received.

## Reproducibility and safe operation

The committed replay is offline and read-only:

```bash
git clone https://github.com/Jennycruzy/convex.git
cd convex
uv venv
uv pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/python -m scripts.replay --session 2026-09-01
```

The profile evaluator needs Alpaca historical data and paper credentials, but remains read-only with respect to trading:

```bash
.venv/bin/python -m scripts.profile_backtest --days 800
.venv/bin/python -m scripts.profile_backtest --days 800 --json data/profile-report.json
```

The live-facing commands are intentionally separate. Use the paper account and verify the dry run before any paper submission:

```bash
cp .env.example .env
.venv/bin/python -m scripts.preflight
.venv/bin/python -m scripts.calibrate_costs
.venv/bin/python -m scripts.run_cycle --dry-run
```

The non-dry-run cycle is paper execution, not a simulator. It is scheduled for the configured market time and is followed by `scripts.manage` and `scripts.manage --settle`. The dashboard reads the ledger; it does not create orders.

## Repository map

- `convex/data/`: MCP transport and typed Alpaca gateway. This is the only market-data and execution boundary.
- `convex/features.py`, `convex/classifier.py`, and `convex/training.py`: point-in-time feature, classifier, and recorded-chain training logic.
- `convex/structures/`, `convex/costs.py`, `convex/edge.py`, `convex/sizing.py`: candidate construction, cost waterfall, tail arithmetic, and deterministic sizing.
- `convex/gates.py`: six session gates plus ten candidate gates; the leg-count item is audit-visible but deliberately non-blocking.
- `convex/agent.py`: the normal decision cycle, ledger receipts, and fresh-quote zero-fill retry path.
- `convex/gap_continuation.py` and `scripts/run_gap_continuation.py`: the isolated one-lot research profile and its in-memory activation boundary.
- `scripts/profile_backtest.py`: profile-specific threshold grid, untouched holdout, and expanding walk-forward evaluator.
- `scripts/replay.py`, `scripts/backtest.py`, `scripts/calibrate_costs.py`, and `scripts/preflight.py`: reproducibility, recorded-chain replay, live calibration, and environment checks.
- `data/ledger/`: append-only decisions and corrections. `data/chains/`, `data/scenarios/`, and `data/variance/` carry the evidence used by research and the live paper path.
- `deploy/`: systemd units and timers for calibration, the session, management, and settlement.
- `docs/WRITEUP.md`: the concise submission narrative; this README is the operational and methodological detail.

## Limitations and promotion criteria

This repository is paper-only and is not investment advice. The SPXW research that motivates the structure choices is not proof about SPY. SPY has American exercise, physical settlement, different strike spacing, different liquidity, and different fees. Reconstructed historical option sessions contain prints but not the original order book. A normal-approximation lower bound is a guardrail, not a guarantee, especially for fat-tailed 0DTE returns. The sample is still too small and unstable to support a positive-return claim.

The profile earns a broader policy only after a new run clears the same gates on genuinely untouched data: sufficient training and holdout trades, positive lower bounds after modeled costs, positive holdout net P&L, and a positive expanding walk-forward result. Until then, a refusal is the correct output.

The submission's strength is therefore falsifiability: the code exposes where a tempting trade loses its edge, makes every risk assumption inspectable, and preserves enough receipts for another reviewer to reproduce the conclusion.
