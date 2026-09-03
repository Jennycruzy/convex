# Chronological P&L audit — 2 September 2026

Purpose: decide whether any current CONVEX family may resume after the live paper-account loss. This was a read-only test: no order, ledger record, or account state was changed.

**Status note (3 September):** this report records the family-screen verdict. Both BWB families remain disabled. The separate one-lot gap-continuation runner is a conditional paper-observation profile added afterward; it does not change the negative BWB results or claim that the gap lead is proven.

Method:

- 178 SPY option sessions reconstructed from Alpaca history.
- Archived 514-session scenario distribution from `2026-09-02 10:00`.
- 1.0% relative spread per leg, the strictest previously considered executable regime.
- First 124 sessions: model expansion and selection history.
- Final 54 sessions beginning 17 June 2026: untouched evaluation segment.
- Baseline: cash / no trade = $0 P&L.

| Family | Training trades | Training net P&L | Held-out trades | Held-out net P&L | Held-out Sharpe | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Put BWB | 28 | +$440.14 | 20 | **−$349.55** | −1.404 | Fail |
| Call BWB | 25 | −$129.08 | 24 | **−$330.95** | −1.484 | Fail |

Neither BWB family beats cash after modeled execution costs on untouched data. Both are disabled in `config/convex.yaml`; the existing kill switch remains engaged.

## Alternative-family screen

The same 125/54 chronological split was then run in memory across the former alternative families. This did not enable or submit any strategy.

| Family | Training trades | Training net P&L | Held-out trades | Held-out net P&L | Verdict |
|---|---:|---:|---:|---:|---|
| Debit vertical | 51 | −$1,448.42 | 20 | −$671.47 | Fail |
| Straddle | 30 | +$550.40 | 11 | −$559.54 | Fail |
| Strangle | 15 | −$637.92 | 6 | +$1,017.86 | Fail: contradictory training result and six-trade holdout |

No candidate passes both the selection and held-out periods. A new family may resume only after it clears this same chronological test with positive held-out net P&L, sufficient observations, and a positive lower confidence bound.

## Trend-conditioned vertical research

A distinct, pre-existing hypothesis was evaluated after the family screen: at 10:00 ET, take a debit vertical only after a 15-minute opening-range break agrees with VWAP by at least 0.05%; use calls for an upside break and puts for a downside break. This is not the earlier variance-regime debit-vertical selector.

| Segment | Trades | Net P&L | Mean P&L | 95% one-sided lower bound | Hit rate |
|---|---:|---:|---:|---:|---:|
| Selection period | 60 | +$431.03 | +$7.18 | −$37.76 | 56.67% |
| Untouched final period | 26 | +$454.34 | +$17.47 | **−$46.36** | 53.85% |

The positive held-out total makes this the first candidate worth further observation, but it does **not** pass the pre-committed admission criterion: the 95% lower confidence bound remains negative. It is therefore research-only and is not enabled for paper execution. Searching thresholds on this same held-out segment would contaminate the test, so any refinement must be selected on an earlier segment and evaluated on a new untouched period.
