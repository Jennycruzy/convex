# SPY 0DTE execution cost, measured 2026-08-29 21:28 UTC

Underlying at 769.30, expiry 2026-08-31, 124 contracts in the band.

**Taken with the market closed.** These are the spreads of a book nobody is quoting. Nothing was written into the configuration from them and nothing should be.

## Per leg

| | median | p90 |
|---|---|---|
| half-spread | 0.025 | 1.430 |
| relative spread | 13.7% | 200.0% |

## Per structure, one contract

| family | candidates | median cost | p90 cost | median cost as a share of the worst case |
|---|---|---|---|---|
| put_bwb | 400 | 20.20 | 544.70 | 9.8% |
| call_bwb | 400 | 28.70 | 350.20 | 13.1% |
| straddle | 1 | 5.35 | 5.35 | 1.4% |
| strangle | 196 | 3.90 | 5.90 | 11.2% |
| debit_vertical | 400 | 11.10 | 151.10 | 12.1% |
