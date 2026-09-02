# SPY 0DTE execution cost, measured 2026-09-02 12:43 UTC

Underlying at 761.85, expiry 2026-09-02, 122 contracts in the band.

**Taken with the market closed.** These are the spreads of a book nobody is quoting. Nothing was written into the configuration from them and nothing should be.

## Per leg

| | median | p90 |
|---|---|---|
| half-spread | 0.025 | 1.540 |
| relative spread | 12.7% | 200.0% |

## Per structure, one contract

| family | candidates | median cost | p90 cost | median cost as a share of the worst case |
|---|---|---|---|---|
| put_bwb | 400 | 113.28 | 510.28 | 36.7% |
| call_bwb | 400 | 78.03 | 408.78 | 45.4% |
| straddle | 1 | 3.59 | 3.59 | 0.9% |
| strangle | 28 | 3.09 | 5.59 | 2.7% |
| debit_vertical | 392 | 22.14 | 111.64 | 5.3% |
