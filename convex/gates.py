"""The twelve risk gates.

Every gate is an object with a name, a scope and a verdict, so the whole set
can be listed on the dashboard and in the write-up whether or not it fired
today. A gate that has never been observed rejecting anything is a gate nobody
has tested, so each one carries the numbers it compared and the ledger keeps
them: the receipt for a refusal is as complete as the receipt for a trade.

Session gates run once per cycle, before any candidate is priced. Candidate
gates run per structure. Two of them are unusual and worth stating plainly:

  leg count      is a preference, not a veto. It never blocks a trade; it
                 records the crossings a structure needs so a two-legged
                 candidate wins a race against a four-legged one at comparable
                 net edge, which is the term the research found decisive.

  assignment     is the SPY-specific one. The research is cash-settled European
                 SPX; these are physically settled American options, so a short
                 leg that is in the money near the close is an assignment
                 waiting to happen and is refused rather than managed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, Sequence

from convex.config import Config
from convex.edge import EdgeEstimate
from convex.errors import ConfigError
from convex.sizing import SizeDecision
from convex.structures.base import Candidate


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict, with the comparison that produced it."""

    name: str
    scope: str
    passed: bool
    detail: str
    observed: float | None = None
    threshold: float | None = None
    blocking: bool = True

    def as_dict(self) -> dict:
        record = {
            "gate": self.name,
            "scope": self.scope,
            "passed": self.passed,
            "detail": self.detail,
            "blocking": self.blocking,
        }
        if self.observed is not None:
            record["observed"] = round(self.observed, 6)
        if self.threshold is not None:
            record["threshold"] = round(self.threshold, 6)
        return record


@dataclass
class GateReport:
    """The full set of verdicts for one cycle or one candidate."""

    results: list[GateResult] = field(default_factory=list)

    def add(self, result: GateResult) -> "GateReport":
        self.results.append(result)
        return self

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results if result.blocking)

    @property
    def failures(self) -> list[GateResult]:
        return [r for r in self.results if r.blocking and not r.passed]

    @property
    def first_failure(self) -> GateResult | None:
        return self.failures[0] if self.failures else None

    def as_dicts(self) -> list[dict]:
        return [result.as_dict() for result in self.results]


@dataclass(frozen=True)
class GateContext:
    """Everything the gates are allowed to look at, gathered once per cycle."""

    config: Config
    now_exchange: datetime
    session_close: datetime
    is_trading_day: bool
    market_open: bool
    equity: float
    last_equity: float
    buying_power: float
    spot: float
    open_structures: int
    es_in_use: float
    cumulative_fees: float
    kill_switch_path: Path
    submission_cutoff: datetime | None = None
    probability: float | None = None

    @property
    def minutes_to_close(self) -> float:
        return (self.session_close - self.now_exchange).total_seconds() / 60.0

    @property
    def day_pnl_pct(self) -> float:
        if self.last_equity <= 0.0:
            raise ConfigError(f"prior equity must be positive, found {self.last_equity}")
        return (self.equity - self.last_equity) / self.last_equity


class Gate(Protocol):
    name: str
    scope: str

    def check(
        self,
        context: GateContext,
        candidate: Candidate | None,
        estimate: EdgeEstimate | None,
        size: SizeDecision | None,
    ) -> GateResult: ...


# ------------------------------------------------------------------ session


class KillSwitchGate:
    name = "kill_switch"
    scope = "session"

    def check(self, context, candidate, estimate, size) -> GateResult:
        engaged = context.kill_switch_path.exists()
        detail = (
            f"kill switch file present at {context.kill_switch_path}"
            if engaged
            else "no kill switch file"
        )
        return GateResult(self.name, self.scope, not engaged, detail)


class CalibrationGate:
    """Refuse to open a position on numbers that have never been measured.

    config/convex.yaml marks a value HYPOTHESIS until it has been measured
    against real SPY quotes, and until now that marking was a comment: nothing
    stopped a live decision reading one. Two of those values are the per-contract
    fees, which stand at zero. Trading on them would understate cost inside the
    net-of-cost hurdle, which is the one number this project exists to get right,
    and it would understate it in the direction that makes a candidate look
    tradeable when it is not.

    Standing down here is the correct outcome, not a failure: it is cheaper to
    trade nothing than to trade on a cost model that was assumed.
    """

    name = "calibration"
    scope = "session"

    # What a live decision actually reads. A value that only a backtest touches
    # does not belong here; the replay is allowed to run on hypotheses so long
    # as it says so, and it does.
    REQUIRED = (
        "costs.slippage_ticks_per_leg",
        "costs.per_contract_fee",
        "costs.regulatory_fee_per_contract",
        "liquidity.max_relative_spread",
        "session.pin_band_pct",
    )

    def check(self, context, candidate, estimate, size) -> GateResult:
        unmeasured = context.config.unmeasured(*self.REQUIRED)
        if unmeasured:
            return GateResult(
                self.name,
                self.scope,
                False,
                "never measured against live quotes: "
                + ", ".join(unmeasured)
                + " — run scripts/calibrate_costs.py while the market is open",
            )
        return GateResult(
            self.name,
            self.scope,
            True,
            "every cost and liquidity input has been measured against live quotes",
        )


class MarketCalendarGate:
    name = "market_calendar"
    scope = "session"

    def check(self, context, candidate, estimate, size) -> GateResult:
        if not context.is_trading_day:
            return GateResult(
                self.name, self.scope, False, "Alpaca's calendar has no session today"
            )
        if not context.market_open:
            return GateResult(self.name, self.scope, False, "the market is not open right now")
        if context.submission_cutoff is not None and context.now_exchange >= context.submission_cutoff:
            return GateResult(
                self.name,
                self.scope,
                False,
                f"past the submission cutoff at {context.submission_cutoff:%Y-%m-%d %H:%M %Z}, "
                "no new positions are opened",
            )
        guard = context.config.float_("session.assignment_guard_minutes")
        if context.minutes_to_close <= guard:
            return GateResult(
                self.name,
                self.scope,
                False,
                "inside the assignment guard window, too late to open",
                observed=context.minutes_to_close,
                threshold=guard,
            )
        return GateResult(
            self.name,
            self.scope,
            True,
            f"session open with {context.minutes_to_close:.0f} minutes to the close",
            observed=context.minutes_to_close,
        )


class DailyLossLimitGate:
    name = "daily_loss_limit"
    scope = "session"

    def check(self, context, candidate, estimate, size) -> GateResult:
        limit = context.config.float_("risk.daily_loss_limit_pct")
        loss = -context.day_pnl_pct
        breached = loss >= limit
        return GateResult(
            self.name,
            self.scope,
            not breached,
            f"day P&L is {context.day_pnl_pct:+.2%} against a {limit:.0%} limit",
            observed=loss,
            threshold=limit,
        )


class BuyingPowerGate:
    name = "buying_power"
    scope = "session"

    def check(self, context, candidate, estimate, size) -> GateResult:
        available = context.buying_power > 0.0
        return GateResult(
            self.name,
            self.scope,
            available,
            f"account reports {context.buying_power:,.2f} of buying power",
            observed=context.buying_power,
        )


class CostBudgetGate:
    name = "cost_budget"
    scope = "session"

    def check(self, context, candidate, estimate, size) -> GateResult:
        budget = context.config.float_("risk.cumulative_fee_budget_pct") * context.equity
        spent = context.cumulative_fees
        return GateResult(
            self.name,
            self.scope,
            spent < budget,
            f"{spent:,.2f} of execution cost paid against a budget of {budget:,.2f}",
            observed=spent,
            threshold=budget,
        )


# ---------------------------------------------------------------- candidate


class MaxLossGate:
    name = "max_loss"
    scope = "candidate"

    def check(self, context, candidate, estimate, size) -> GateResult:
        _require(candidate, estimate)
        budget = context.equity * context.config.float_("risk.risk_pct_per_structure")
        worst = estimate.profile.max_loss
        return GateResult(
            self.name,
            self.scope,
            worst <= budget,
            f"worst case {worst:,.2f} per lot at {estimate.profile.max_loss_price:g} "
            f"against a per-structure budget of {budget:,.2f}",
            observed=worst,
            threshold=budget,
        )


class NetOfCostGate:
    name = "net_of_cost"
    scope = "candidate"

    def check(self, context, candidate, estimate, size) -> GateResult:
        _require(candidate, estimate)
        net = estimate.net_edge
        passed = net > 0.0
        if estimate.gross_edge > 0.0 and not passed:
            detail = (
                f"gross edge {estimate.gross_edge:,.2f} is entirely consumed by "
                f"{estimate.cost.total:,.2f} of execution cost across "
                f"{estimate.cost.leg_count} legs, leaving {net:,.2f}"
            )
        else:
            detail = f"net edge {net:,.2f} after {estimate.cost.total:,.2f} of cost"
        return GateResult(self.name, self.scope, passed, detail, observed=net, threshold=0.0)


class LegCountPreference:
    name = "leg_count"
    scope = "candidate"

    def check(self, context, candidate, estimate, size) -> GateResult:
        _require(candidate, estimate)
        legs = estimate.cost.leg_count
        return GateResult(
            self.name,
            self.scope,
            True,
            f"{legs} legs to cross, costing {estimate.cost.half_spread:,.2f} in half-spread",
            observed=float(legs),
            blocking=False,
        )


class LiquidityGate:
    name = "liquidity"
    scope = "candidate"

    def check(self, context, candidate, estimate, size) -> GateResult:
        _require(candidate, estimate)
        max_relative = context.config.float_("liquidity.max_relative_spread")
        min_size = context.config.int_("liquidity.min_displayed_size")
        worst_symbol = ""
        worst_relative = 0.0
        for leg in candidate.legs:
            quote = leg.entry.quote
            if quote.relative_spread > worst_relative:
                worst_relative, worst_symbol = quote.relative_spread, quote.symbol
            displayed = quote.ask_size if leg.ratio > 0 else quote.bid_size
            if displayed < min_size:
                return GateResult(
                    self.name,
                    self.scope,
                    False,
                    f"{quote.symbol} shows {displayed} contracts on the touch, "
                    f"below the {min_size} required",
                    observed=float(displayed),
                    threshold=float(min_size),
                )
        return GateResult(
            self.name,
            self.scope,
            worst_relative <= max_relative,
            f"widest leg is {worst_symbol} at {worst_relative:.1%} of mid "
            f"against a {max_relative:.1%} limit",
            observed=worst_relative,
            threshold=max_relative,
        )


class ExpectedShortfallGate:
    name = "expected_shortfall"
    scope = "candidate"

    def check(self, context, candidate, estimate, size) -> GateResult:
        _require(candidate, estimate)
        if size is None:
            raise ConfigError("the expected shortfall gate needs a size to project against")
        cap = context.equity * context.config.float_("risk.portfolio_es_cap_pct")
        projected = context.es_in_use + estimate.expected_shortfall * size.contracts
        return GateResult(
            self.name,
            self.scope,
            projected <= cap,
            f"portfolio one-percent tail would be {projected:,.2f} against a cap of {cap:,.2f}",
            observed=projected,
            threshold=cap,
        )


class AssignmentGate:
    name = "assignment"
    scope = "candidate"

    def check(self, context, candidate, estimate, size) -> GateResult:
        _require(candidate, estimate)
        guard = context.config.float_("session.assignment_guard_minutes")
        itm_shorts = [
            leg.contract.symbol
            for leg in candidate.short_legs
            if leg.contract.is_itm(context.spot)
        ]
        if itm_shorts and context.minutes_to_close <= guard:
            return GateResult(
                self.name,
                self.scope,
                False,
                f"{len(itm_shorts)} short leg(s) are in the money with "
                f"{context.minutes_to_close:.0f} minutes to the close: "
                "these are American-style options and would be an assignment risk",
                observed=context.minutes_to_close,
                threshold=guard,
            )
        return GateResult(
            self.name,
            self.scope,
            True,
            f"{len(itm_shorts)} in-the-money short leg(s) with "
            f"{context.minutes_to_close:.0f} minutes to the close",
            observed=float(len(itm_shorts)),
        )


class ClassifierConfidenceGate:
    name = "classifier_confidence"
    scope = "candidate"

    def check(self, context, candidate, estimate, size) -> GateResult:
        _require(candidate, estimate)
        if context.probability is None:
            raise ConfigError("the confidence gate needs a probability for this structure")
        band = context.config.float_("classifier.confidence_band")
        distance = abs(context.probability - 0.5)
        return GateResult(
            self.name,
            self.scope,
            distance > band and context.probability > 0.5,
            f"probability of a profitable outcome is {context.probability:.3f}, "
            f"{distance:.3f} from the coin flip against a band of {band:.3f}",
            observed=context.probability,
            threshold=0.5 + band,
        )


class FeatureStalenessGate:
    name = "feature_staleness"
    scope = "candidate"

    def check(self, context, candidate, estimate, size) -> GateResult:
        _require(candidate, estimate)
        budget = context.config.float_("liquidity.max_quote_age_seconds")
        oldest = max(leg.entry.quote.age_seconds(context.now_exchange) for leg in candidate.legs)
        return GateResult(
            self.name,
            self.scope,
            oldest <= budget,
            f"oldest leg quote is {oldest:.1f}s old against a {budget:.0f}s budget",
            observed=oldest,
            threshold=budget,
        )


class ConcurrencyGate:
    name = "concurrency"
    scope = "candidate"

    def check(self, context, candidate, estimate, size) -> GateResult:
        _require(candidate, estimate)
        cap = context.config.int_("risk.max_concurrent_structures")
        return GateResult(
            self.name,
            self.scope,
            context.open_structures < cap,
            f"{context.open_structures} structures open against a cap of {cap}",
            observed=float(context.open_structures),
            threshold=float(cap),
        )


def _require(candidate: Candidate | None, estimate: EdgeEstimate | None) -> None:
    if candidate is None or estimate is None:
        raise ConfigError("a candidate gate was run without a candidate and a priced estimate")


SESSION_GATES: tuple[Gate, ...] = (
    KillSwitchGate(),
    CalibrationGate(),
    MarketCalendarGate(),
    DailyLossLimitGate(),
    BuyingPowerGate(),
    CostBudgetGate(),
)

CANDIDATE_GATES: tuple[Gate, ...] = (
    MaxLossGate(),
    NetOfCostGate(),
    LegCountPreference(),
    LiquidityGate(),
    ExpectedShortfallGate(),
    AssignmentGate(),
    ClassifierConfidenceGate(),
    FeatureStalenessGate(),
    ConcurrencyGate(),
)

ALL_GATES: tuple[Gate, ...] = SESSION_GATES + CANDIDATE_GATES


def run_session_gates(context: GateContext) -> GateReport:
    report = GateReport()
    for gate in SESSION_GATES:
        report.add(gate.check(context, None, None, None))
    return report


def run_candidate_gates(
    context: GateContext,
    candidate: Candidate,
    estimate: EdgeEstimate,
    size: SizeDecision,
    gates: Sequence[Gate] = CANDIDATE_GATES,
) -> GateReport:
    report = GateReport()
    for gate in gates:
        report.add(gate.check(context, candidate, estimate, size))
    return report
