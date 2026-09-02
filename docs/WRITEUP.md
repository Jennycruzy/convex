# CONVEX · one-page write-up

**Alpaca account ID:** `PA35QSFNW15J` (paper, opened 2026-08-29 00:20 ET at exactly $100,000.00, options level 3). Brand new and dedicated to this hackathon, opened after the 28 August kick off and never used for anything else.
**Repo:** github.com/Jennycruzy/convex · **Demo:** https://convex.isobars.xyz
**Instrument:** SPY 0DTE options, paper account, entry 10:00 ET, held to the 16:00 close.

---

## The claim

Most 0DTE bots sell iron condors or buy butterflies. **"0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing"** (SPXW, 09/2016–01/2026; replication package at `github.com/vilkovgr/0dte-strategies`) tested both across ten years net of realistic frictions and found the iron butterfly/condor family at **gross Sharpe 0.77, net −0.20**. A long butterfly and an iron butterfly are synthetically equivalent, so that is the structure most of this field ships. It does not fail because the payoff shape is wrong. It fails because four legs of bid/ask consume the edge.

CONVEX is built on the parts of that paper that survived costs: **put ratio spreads (gross 1.18 / net 0.93)**, **strangles and straddles (0.56 / 0.39)**, and the finding that a **basket across families beats any single structure (1.12 / 0.82)**.

## The AI logic

**Directional classification, not return prediction.** The paper is explicit that conditional timing works better as binary classification than as magnitude forecasting, and that within binary designs hard mapping dominates soft mapping. So for each structure family *s* on day *t*:

```
y[s,t] = 1 if net PnL > 0 else 0
p[s,t] = LogisticRegression(L2).predict_proba(X[s,t])
w[s,t] = sign(p[s,t] − 0.5)        # full size or nothing
```

**L2 logistic regression, deliberately, and not a limitation.** The paper found a low-variance parametric model beats higher-capacity ones on short-horizon 0DTE data with noisy payoffs and small per-strategy samples. There is no neural network here on purpose, and the model is auditable as a result: every coefficient is inspectable and every probability is reproducible from the recorded feature row.

**Features, all observable by 10:00 ET, no look-ahead.** Integrated implied variance from the 0DTE chain (VIX methodology, 10:00 to 16:00); **implied skew `IS = IV_up − IV_dn`**, the highest-value feature, because realized skewness drives 0DTE PnL more than realized variance does; slope proxies; lagged realized moments; each family's own lagged results; and liquidity terms (half-spread, depth, relative spread, tightness) which feed cost directly. Alpaca currently omits Greeks and open interest on expiring SPY contracts, so the GEX-style exposure proxy is displayed only when supplied and is not a classifier input. Predictors are standardised inside the training window only, on an expanding window.

**Honest caveat, stated because it is true:** SPY 0DTE history available through Alpaca is short relative to the burn-in a walk-forward protocol wants. Where it is insufficient the agent says so and runs a documented volatility-regime rule instead: downside structures in high-implied-variance states, upside in low. The ledger records which of the two made every single call. Hit rate, Brier score and calibration slope are reported, not accuracy alone.

**The LLM computes nothing.** Featherless narrates a brief of figures already computed by deterministic code. A numerical-token guard rejects any narration that introduces, changes, or re-rounds a number; the deterministic rationale is written instead and the record says which one it was.

## The structures

**S1 put broken-wing butterfly (primary).** The paper's best structure is the put ratio spread, but a raw ratio spread has an open downside. A 1×2 put ratio plus a protective lower wing *is* a put broken-wing butterfly: the same skew exposure with the risk defined, usually entered for a credit, so above every strike it keeps the credit and risks nothing. **S2** call BWB (weaker per the paper, so the classifier decides). **S3** long straddle/strangle. **S4** debit vertical, two legs instead of four, which matters enormously under cost. **S5 stand down**, a first-class logged outcome.

**Not built: symmetric butterflies and iron condors.** Negative net Sharpe. Declining to build them is the thesis.

## The risk gates

Fifteen, every one logged with its verdict whether it passes or fails.

*Session scope:* **1** kill switch (append-only file, every cycle) · **2** **calibration: no position is opened while any cost or liquidity input is still an unmeasured guess.** The live spread threshold is measured at 09:55 ET and a fresh successful receipt is required for the exact 10:00 ET entry. Fee assumptions are conservative bounds until Alpaca's day-level activity provides actual fee receipts · **3** market calendar, from Alpaca's own endpoint rather than a local holiday table · **4** daily loss limit, 3% of equity then halt and publish · **5** buying power, verified against the account · **6** cumulative cost budget, 2% of equity.

*Candidate scope:* **7** max loss computable and within budget: the agent is structurally incapable of submitting a position whose worst case it cannot compute, because pricing runs *through* the max-loss calculator · **8** **net-of-cost hurdle: edge must exceed half-spread × legs + slippage. The most important gate in the system** · **9** leg-count preference, two legs over four at comparable net edge · **10** liquidity, rejecting any leg whose relative spread exceeds the measured threshold · **11** portfolio ES(1%) cap at 3% of equity · **12** **assignment: no leg that can settle into shares survives the final thirty minutes** · **13** classifier confidence, standing down when probabilities cluster at 0.5 · **14** feature staleness · **15** concurrency, at most four open structures.

**Sizing is computed, never chosen.** `risk_budget = equity × 1%`; `contracts = floor(risk_budget / max_loss)`; then the portfolio ES check. One function, no override parameter. Confidence decides *whether* to trade and never *how much*, because ES(1%) at 0DTE runs roughly 0.58–1.58% of underlying, which makes mean PnL an inadequate summary statistic on its own.

**Positions are held to expiry.** No stop losses on defined-risk structures: stopping out converts a capped loss into a realised loss plus multi-leg slippage. Exactly three things close early: kill switch, daily loss limit, assignment guard.

## The Alpaca infrastructure

**MCP is the entire data and execution layer.** The agent's only route to the market is the server Alpaca publishes (`alpaca-mcp-server`), spawned as a child process over stdio, with the toolset restricted to account, trading, assets, stock-data and options-data. Chains, Greeks, implied volatility, open interest, quotes, the market calendar, positions, orders: all of it arrives over MCP. Structures go out as **atomic multi-leg limit orders**, so every leg fills together or none does, which removes partial-fill risk from a four-legged position. Never a market order: a market order on a spread is a donation to the other side.

Two failure modes get explicit handling because they would otherwise be silent. The order tools answer with a *successful* JSON-RPC result whose body can carry an error object, so every payload is inspected before it can be read as a fill. And every call carries a timeout, because a cycle that hangs at 10:00 is a cycle that misses its entry. Response parsing is written against the OpenAPI specs shipped inside the server package; a field the agent depends on raises when absent rather than becoming a zero.

**SPX research, SPY execution.** Alpaca lists equity and ETF options, not index options, so every parameter was re-derived: $1 strikes instead of $5, **physical American settlement instead of cash European**, ~$650 underlying instead of ~6,800, and the paper's ±2% moneyness band mapping to roughly ±$13. Widths are expressed as fractions of spot, never as absolute points. Physical settlement is why the assignment guard exists at all. It has no counterpart in the paper being implemented, and it closes shorts before longs so no ordering ever leaves an undefined-risk position standing.

**Every decision leaves a receipt.** An append-only JSONL ledger records timestamp, per-structure probability and its source, the feature row, strikes, net debit or credit, the full cost breakdown, max loss, ES contribution, size and its binding constraint, every gate verdict, the written rationale, and the outcome, **including every refusal and every stand-down**. The public dashboard is a view over that ledger and computes nothing of its own, so the page a judge loads and the evidence in this write-up are the same artefact.

## What this is not

Alpaca's calendar puts six sessions in this window (Aug 28, 31, Sep 1, 2, 3, 4; Labor Day is Sep 7, not Sep 1), and CONVEX will trade fewer than that. A handful of sessions is not a track record; P&L across this window is substantially variance. Alpaca currently omits Greeks and open interest on expiring SPY contracts, so GEX-style exposure is an optional displayed proxy rather than a classifier input or a dealer-inventory reconstruction. The live spread threshold is recalibrated from real SPY quotes immediately before entry and broker fees are reconciled from Alpaca activity after fills. The reproducible contribution here is the cost discipline, the risk gates, and the receipts, not the P&L.
