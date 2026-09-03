# Build-in-public posts

Post these from your own X or LinkedIn account and tag **@lablabai** and **@AlpacaHQ** (and their LinkedIn pages). Replace the dashboard URL only if it changes.

## Post 1 — the problem

Most 0DTE option bots die in the bid/ask spread, not in the payoff diagram.

I’m building CONVEX for the Alpaca AI Trading Agents Hackathon: an autonomous SPY options agent that rejects trades unless the edge survives spreads, slippage, fees, and a conservative exit reserve.

It uses Alpaca MCP end-to-end and publishes every refusal—not just its wins. Demo: https://convex.isobars.xyz

#Alpaca #lablabai #AITRADING #Options

## Post 2 — the technical insight

A quote can be executable and still be unprofitable.

CONVEX’s replay stayed positive through a 1.0% relative spread per option leg and turned negative at 1.5%. So I made 1.0% the live admission ceiling. Wide books don’t trigger “try harder”; they trigger a logged refusal.

That is the difference between an expected-P&L chart and an executable strategy. Built with Alpaca MCP for the @lablabai × @AlpacaHQ hackathon.

## Post 3 — execution integrity

I found and corrected a paper-trading accounting failure: a canceled, zero-fill options order had been allowed to affect reported P&L.

The fix is now architectural: only a broker-verified full fill can create a position or change P&L; corrections are append-only; multi-leg positions close atomically.

Autonomy without a receipt is just a story. CONVEX’s receipts are public: https://convex.isobars.xyz

#Alpaca #lablabai #AIagents

## Final post — demo

Shipped: CONVEX, an autonomous SPY 0DTE options agent for the Alpaca AI Trading Agents Hackathon.

- Alpaca MCP for market data and paper execution
- defined-risk atomic multi-leg orders
- $25 net-edge floor
- positive 95% lower-confidence-bound requirement
- 1.0% per-leg spread ceiling
- public ledger of trades *and* refusals

Demo: https://convex.isobars.xyz
Repo: https://github.com/Jennycruzy/convex

@lablabai @AlpacaHQ
