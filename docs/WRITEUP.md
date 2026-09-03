# CONVEX — a 0DTE options agent that publishes its own falsification

**Alpaca paper account:** PA35QSFNW15J · **Demo:** https://convex.isobars.xyz · **Repo:** https://github.com/Jennycruzy/convex

## The claim

We built the obvious 0DTE trade, measured it honestly, and it failed. That result is the submission.

On the 1 September SPY chain, **482 of 1,101 priced candidates — 43.8% — showed a gross profit that execution cost consumed entirely.** They never reached a risk check; the ranking demoted them first. That is the number this project exists to make visible, and it is why most of what looks profitable at mid-price is not.

CONVEX trades SPY 0DTE defined-risk structures through Alpaca's MCP server, and writes every decision — including every refusal — to an append-only ledger.

## Reproduce this

```
git clone https://github.com/Jennycruzy/convex && cd convex
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m scripts.replay --session 2026-09-01
```

Reads only files in the repository. No network, no credentials, no order, no ledger write. The recorded chains and the scenario sets are committed, so the figure above reproduces from a clean clone. The same command on 31 August gives 554 of 1,024, or 54.1%.

## What our own model did

Five structure families were trained and scored against a majority-class baseline on 509 reconstructed sessions. **One cleared the skill bar: debit_vertical, at 0.0512 against the 0.0470 required. It still lost $12.87 per session.** The other four did not clear it at all. Nothing was promoted on the strength of that classifier.

A chronological audit on 2 September re-tested every family on a 54-session segment never used for selection. All five failed after costs: put BWB −$349.55, call BWB −$330.95, debit vertical −$671.47, straddle −$559.54. Strangle returned +$1,017.86 on six trades, which is not evidence. All five families are disabled in config: `structures.enabled` is empty, so none of them can open new risk.

One hypothesis survives as research only — a trend-conditioned vertical, +$454.34 over 26 untouched trades — and it is **not enabled**, because its 95% lower confidence bound is −$46.36. Positive total, negative bound, no trade. Searching thresholds on that same segment would contaminate it.

## What it actually did

Three verified fills, reconciled against broker records, not inferred from submitted orders:

| Date | Structure | Lots | Net P&L |
|---|---|---:|---:|
| 1 Sep | put_bwb | 14 | −$565.30 |
| 1 Sep | straddle | 7 | −$281.32 |
| 2 Sep | call_bwb | 6 | −$180.00 |

**Total −$1,026.62, or −1.03% on the $100,000 account.**

The third trade needs disclosing. On 2 September a tournament profile was authorized to submit one BWB structure (ledger receipt 102) after the audit had already disabled that family. It filled six contracts and lost $180. The profile has since been removed. We are reporting this because the ledger records it and a reader can find it; the alternative is a write-up the receipts contradict.

The ledger also carries a correction receipt (145) retracting a −$840 expiry P&L for an entry Alpaca had canceled at 0/10 filled. Errors are corrected in the append-only record, never overwritten.

## Risk checks

Every cycle runs six session gates — kill switch, calibration provenance, market calendar, daily loss limit, buying power, cost budget — and every surviving candidate runs ten more: max loss computed before submission, net-of-cost edge, positive 95% lower bound on scenario mean net P&L, leg-count preference, per-leg liquidity and relative-spread admission, 1% expected shortfall, assignment guard, classifier confidence, feature staleness, and single-position concurrency. A candidate must clear all sixteen. Standing down is a successful outcome, and the refusal is written with its arithmetic.

## Infrastructure

Alpaca MCP is the only market-data and execution route: account, calendar, contracts, quotes, orders, positions, paper fills. Entries and assignment exits are atomic multi-leg limit orders — the full defined-risk structure fills or a non-fill is recorded. No market orders, no leg-by-leg disassembly. The LLM narrates figures already computed; it cannot produce a price, Greek, probability, size, or order, and the agent trades normally with it switched off.

## Attribution

Structure selection follows *"0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing"* (SPXW, 09/2016–01/2026). The paper reports a gross Sharpe of 0.77 against a net Sharpe of −0.20 for the iron butterfly/condor, and 1.18 against 0.93 for the put ratio spread. Those are the paper's measurements, not ours. Our own basket, priced over 509 reconstructed sessions, went from a gross Sharpe of 1.93 to a net 0.40 on selected sessions, and from 0.20 to −1.94 when every session is traded.

**Paper-trading disclosure:** paper environment only. Results are hypothetical and are not investment advice.
