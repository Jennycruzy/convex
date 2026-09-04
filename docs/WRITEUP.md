# CONVEX — an execution-grade 0DTE agent for durable net P&L

**Alpaca paper account:** PA35QSFNW15J · **Demo:** https://convex.isobars.xyz · **Repo:** https://github.com/Jennycruzy/convex

## The thesis

CONVEX compounds net P&L by refusing trades whose apparent edge disappears after execution cost, tail exposure, or assignment risk.

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

A chronological audit on 2 September re-tested the generic families on a 54-session segment never used for selection. Put BWB returned −$349.55, call BWB −$330.95, debit vertical −$671.47, and straddle −$559.54 after costs. Strangle returned +$1,017.86 on six trades, which is not evidence. Those generic families remain disabled in config: `structures.enabled` is empty. The separate gap runner can expose only one debit vertical in memory after its deterministic signal; it does not promote the generic family.

The gap-continuation vertical is still research evidence, not a proven winner: the small sample was +$454.34 over 26 trades, but its one-sided 95% lower confidence bound was −$46.36. The profile is constrained to one lot, runs in paper only, and remains subject to the untouched holdout and walk-forward promotion bar. Positive total, negative bound, no promotion.

## What it actually did

Three verified fills, reconciled against broker records, not inferred from submitted orders. The date is the session the structure was entered and expired; the two 31 August structures were reconciled the following morning, once the broker's own fill records were available.

| Entered | Structure | Lots | Net P&L |
|---|---|---:|---:|
| 31 Aug | put_bwb | 14 | −$565.30 |
| 31 Aug | straddle | 7 | −$281.32 |
| 2 Sep | call_bwb | 6 | −$180.00 |

Those three sum to **−$1,026.62**, net of the $6.62 of day-level broker fees allocated at reconciliation. The account itself closed at **$98,970.95** against the $100,000 it opened with, so the figure a judge reads off Alpaca is **−$1,029.05, or −1.03%**.

The $2.43 between the two is the 2 September fee bill: $1.20 OCC, $0.72 ORF, $0.42 REG, $0.07 TAF, $0.02 CAT, total broker fees for the account $9.05. Those rows carry broker timestamps from earlier that evening but were not returned by the activities query the reconciler ran at 22:09 UTC, so receipt 147 allocated them as zero. Both numbers are stated rather than the one that flatters, because the ledger total and the account balance are different quantities and a reader can check each against the source.

The third trade needs disclosing. On 2 September a tournament profile was authorized to submit one BWB structure (ledger receipt 102) after the audit had already disabled that family. It filled six contracts and lost $180. That one-off BWB profile has since been removed; it is not the current gap-continuation runner. We are reporting this because the ledger records it and a reader can find it; the alternative is a write-up the receipts contradict.

The ledger also carries a correction receipt (145) retracting a −$840 expiry P&L for an entry Alpaca had canceled at 0/10 filled. Errors are corrected in the append-only record, never overwritten.

## Risk checks

Every cycle runs six session gates — kill switch, calibration provenance, market calendar, daily loss limit, buying power, and cost budget — and every surviving candidate runs ten more: max loss computed before submission, net-of-cost edge, positive 95% lower bound on scenario mean net P&L, leg-count preference, per-leg liquidity and relative-spread admission, 1% expected shortfall, assignment guard, classifier confidence, feature staleness, and single-position concurrency. All blocking checks must pass; leg count is deliberately non-blocking but remains visible in the receipt. Standing down is a successful outcome, and the refusal is written with its arithmetic.

## Infrastructure

Alpaca MCP is the only market-data and execution route: account, calendar, contracts, quotes, orders, positions, paper fills. Entries and assignment exits are atomic multi-leg limit orders — the full defined-risk structure fills or a non-fill is recorded. No market orders, no leg-by-leg disassembly. The LLM narrates figures already computed; it cannot produce a price, Greek, probability, size, or order, and the agent trades normally with it switched off.

## Attribution

Structure selection follows *"0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing"* (SPXW, 09/2016–01/2026). The paper reports a gross Sharpe of 0.77 against a net Sharpe of −0.20 for the iron butterfly/condor, and 1.18 against 0.93 for the put ratio spread. Those are the paper's measurements, not ours. Our own basket, priced over 509 reconstructed sessions, went from a gross Sharpe of 1.93 to a net 0.40 on selected sessions, and from 0.20 to −1.94 when every session is traded.

**Paper-trading disclosure:** paper environment only. Results are hypothetical and are not investment advice.

## The replacement profile

The profile-specific evaluator is research-only: it rebuilds sessions from historical option prints, uses point-in-time stock bars, models a declared per-leg spread, and never writes the live ledger or configuration. An 800-day audited run produced 551 reconstructed sessions, 548 usable observations, 488 training sessions, and a 60-session untouched holdout.

| Test | Net P&L | Trades | 95% lower bound |
|---|---:|---:|---:|
| Configured baseline, training | −$2,407.44 | — | −$132.47 |
| Configured baseline, holdout | +$122.73 | 3 | −$413.38 |
| Best raw training threshold | +$304.92 | 26 | −$93.18 |
| Expanding walk-forward path | −$2,284.71 | 50 | −$125.42 |

No threshold cleared promotion. The small earlier sample (+$454.34 over 26 trades, 95% lower bound −$46.36) remains a hypothesis, not a profitability claim. The active profile stays one lot and paper-only until sufficient training and untouched holdout trades clear positive lower bounds after modeled costs and the expanding walk-forward path is positive.
