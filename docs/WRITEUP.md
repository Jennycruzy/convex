# CONVEX — Cost-Gated Autonomous Options Agent

**Alpaca paper account:** `PA35QSFNW15J` · **Demo:** https://convex.isobars.xyz · **Repository:** https://github.com/Jennycruzy/convex

CONVEX is an autonomous SPY 0DTE options agent built for this hackathon’s dedicated $100,000 Alpaca paper account. It uses Alpaca’s MCP server as its only market-data and execution route, trades defined-risk multi-leg options structures, and publishes every decision—including refusals—as an append-only receipt.

## The problem

A trade that looks profitable at mid-price can be negative after the spread, slippage, fees, and the cost of exit. Most options agents show the first number. CONVEX makes the second number the only one that can authorize an order.

The put and call broken-wing butterfly families are **disabled**. A chronological audit on 2 September found both negative on the untouched final 54-session segment after costs, so they cannot create new risk. For the final paper sessions, the installed runner contains one separate, bounded observation profile: after a qualifying opening gap remains on the same side of VWAP at 10:00 ET, it may enable one direction-matched debit vertical in memory. That profile is still subject to the full net-cost, lower-bound, liquidity, loss, and execution gates, and it may stand down. The project refuses to convert an in-sample idea into a paper trade merely to manufacture activity.

## Autonomous decision logic

At 10:00 ET, the agent obtains the SPY option chain, account, clock, calendar, positions, and quotes from Alpaca MCP. It builds implied-variance, skew, slope, lagged-return, liquidity, and cost features observable at that time. The normal path uses a regularized logistic classifier; the final-day gap profile supplies its documented held-out lead and restricts the candidate direction, but it does not bypass any gate.

The agent then enumerates real contracts and simulates their expiry payoff across its scenario set. It trades only if all of these pass:

- at least **$25** expected net edge after measured spread, slippage, fees, and exit reserve;
- a **positive one-sided 95% lower confidence bound** for the scenario mean net P&L;
- every leg is at or below the lower of the observed threshold and the validated **1.0% relative-spread admission cap**. The replay is positive through 1.0% and negative at 1.5%, so wider quotes are executable but not admissible;
- maximum loss is computed before order submission; one-percent expected shortfall, daily loss, buying power, freshness, and assignment gates all pass;
- at most **one** attributable multi-leg position is open.

The lower-confidence-bound and spread gates are specifically designed to reject an attractive-looking expected value that is too uncertain or too expensive to execute. Standing down is a successful autonomous outcome, not a missing decision.

## P&L evidence and execution integrity

A chronological 178-session audit over a 124-session selection period and untouched 54-session test period invalidated both BWB families after costs: put BWB was −$349.55 on the held-out segment and call BWB was −$330.95. They are disabled. A separate gap-continuation reconstruction produced +$757.37 over 12 held-out verticals, but its historical confidence bound is still negative, so it is treated as a one-lot observation lead—not proven alpha. The live candidate must independently clear the positive lower-bound gate at the current quote. The public dashboard displays the sensitivity sweep and the verified account receipts rather than presenting an in-sample replay as a contest-P&L forecast.

P&L is never inferred from a submitted order. CONVEX credits or debits the dashboard only after Alpaca reports a complete fill. A canceled zero-fill order cannot become a position, settle at expiry, influence model history, or change the dashboard; an append-only correction receipt documents the earlier accounting error rather than hiding it.

Entries and assignment-driven exits are atomic Alpaca multi-leg limit orders: either the full defined-risk structure fills or the agent records a non-fill. It never uses a market order or disassembles a structure leg-by-leg. A close can always occur for risk management; no ordinary strategy action needs human per-order approval.

## Alpaca implementation and presentation

Alpaca MCP is the complete brokerage interface: account, calendar, contracts, quotes, orders, positions, and paper execution all go through its structured tools. The dashboard is server-rendered with no external data dependency and reads directly from the append-only JSONL evidence. Judges can see the selected structure, net-of-cost waterfall, payoff, every gate verdict, order outcome, realized P&L, and every refusal at the demo URL.

The creative contribution is not another opaque “multi-agent hedge fund.” It is an **execution-truth agent**: model ideas are allowed to compete, but deterministic evidence can veto them; the ledger preserves the veto; and P&L remains tied to verified broker facts. The LLM may explain a decision, but cannot calculate a price, probability, size, or order.

**Paper-trading disclosure:** This project operates only in Alpaca’s paper environment. Results are hypothetical, may differ from live execution, and are not investment advice.
