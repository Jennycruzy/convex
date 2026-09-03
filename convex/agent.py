"""The decision cycle.

One pass, in this order, every time:

  1  read the account, the clock and the calendar from Alpaca
  2  run the session checks; a failure ends the cycle and is recorded
  3  take the chain snapshot and build the feature row
  4  ask each family's classifier, or the documented rule, for a probability
  5  enumerate that family's candidates and price every one net of cost
  6  rank on net edge, break ties towards fewer legs
  7  size the best one against the worst case and the tail
  8  run the candidate checks and record every verdict
  9  write the rationale, then send the order, then record the fill

Steps 5 through 9 run for every family the classifier flags, equally weighted,
because the research found a basket across families beat every single structure
it tested. Nothing here concentrates into the highest-probability structure.

Standing down is an outcome of this function, not an error in it. A cycle that
opens nothing and explains why in twelve receipts has done its job.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

from convex import archive
from convex import features as feature_engine
from convex import rationale as rationale_layer
from convex.classifier import RegimeRule, StructureModel
from convex.config import Config
from convex.costs import CostModel
from convex.data.alpaca import AlpacaGateway
from convex.edge import EdgeEstimate, at_limit, evaluate
from convex.errors import ConvexError, DataError, UndefinedRiskError
from convex.execution import filled_quantity, outcome_fields, resolve_order
from convex.gates import GateContext, GateReport, run_candidate_gates, run_session_gates
from convex.instruments import ChainEntry, Leg
from convex.ledger import Action, Ledger, Record, new_cycle_id
from convex.scenarios import ScenarioSet
from convex.sizing import PortfolioState, SizeDecision, size_position
from convex.structures import build_candidates
from convex.structures.base import Candidate, Family


@dataclass
class CycleResult:
    """What one pass did, for the caller and for the dashboard."""

    cycle_id: str
    stood_down: bool
    reason: str
    orders: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class PricedCandidate:
    """A candidate with its single-lot estimate, ready to be ranked."""

    candidate: Candidate
    estimate: EdgeEstimate


def cost_consumed(priced: Sequence[PricedCandidate]) -> list[PricedCandidate]:
    """Candidates that were profitable gross and are not profitable net.

    These never reach a gate. The ranking below is ordered by net edge and the
    walk only ever reaches its first few entries, so a structure whose whole
    gross edge goes to the cost of crossing is demoted out of contention and
    never seen again. That is Law 7 doing the most important thing this agent
    does, and it left no receipt: the net-of-cost gate can only ever fire on a
    candidate cost has already vindicated, which reads as though cost never
    bit. Counting them where they die is what puts the refusal on the record.
    """
    return [
        item
        for item in priced
        if item.estimate.gross_edge > 0.0 and item.estimate.net_edge <= 0.0
    ]


def rank(priced: Sequence[PricedCandidate], tie_break: Sequence[str]) -> list[PricedCandidate]:
    """Best net edge first; a close race goes to the structure with fewer legs.

    The tie band is a percentage of the leader's net edge rather than a fixed
    number of dollars, so it means the same thing on a wide day as on a quiet
    one. Within the band, fewer legs wins, and only then does the family order
    from the configuration break the remaining tie.
    """
    if not priced:
        return []
    order = {name: index for index, name in enumerate(tie_break)}
    leader = max(candidate.estimate.net_edge for candidate in priced)
    band = abs(leader) * _TIE_BAND

    def key(item: PricedCandidate) -> tuple:
        near_leader = leader - item.estimate.net_edge <= band
        return (
            0 if near_leader else 1,
            item.estimate.cost.leg_count if near_leader else 0,
            order.get(str(item.candidate.family), len(order)),
            -item.estimate.net_edge,
        )

    return sorted(priced, key=key)


# Two candidates whose net edges differ by less than this share of the leader's
# are treated as a tie, and the cheaper one to execute wins.
_TIE_BAND = 0.10


class Agent:
    """Wires the pieces together and owns one decision cycle."""

    def __init__(
        self,
        gateway: AlpacaGateway,
        config: Config,
        ledger: Ledger,
        scenarios: ScenarioSet,
        models: dict[Family, StructureModel] | None = None,
        submission_cutoff: datetime | None = None,
        dry_run: bool = False,
        candidate_filter: Callable[[Candidate], bool] | None = None,
        receipt_context: dict | None = None,
        reprice_ticks: tuple[int, ...] = (),
        decision_probability: float | None = None,
        decision_source: str | None = None,
    ) -> None:
        self.gateway = gateway
        self.config = config
        self.ledger = ledger
        self.scenarios = scenarios
        self.models = models or {}
        self.cost_model = CostModel.from_config(config)
        self.rule = RegimeRule()
        self.submission_cutoff = submission_cutoff
        self.dry_run = dry_run
        self.candidate_filter = candidate_filter
        self.receipt_context = receipt_context or {}
        self.reprice_ticks = tuple(tick for tick in reprice_ticks if tick > 0)
        if decision_probability is not None and not 0.0 <= decision_probability <= 1.0:
            raise DataError("decision_probability must lie between zero and one")
        self.decision_probability = decision_probability
        self.decision_source = decision_source
        self.max_attempts = config.int_("candidates.max_ranked_attempts")
        self.zone = ZoneInfo(config.str_("session.timezone"))

    # ------------------------------------------------------------------ inputs

    def _session_close(self, now: datetime) -> datetime:
        sessions = self.gateway.sessions(now.date(), now.date())
        if not sessions:
            raise DataError(f"Alpaca's calendar has no session on {now.date()}")
        close = sessions[0].close_at
        return close if close.tzinfo else close.replace(tzinfo=self.zone)

    def _chain(self, spot: float, expiry) -> list[ChainEntry]:
        low = spot * self.config.float_("candidates.moneyness_low")
        high = spot * self.config.float_("candidates.moneyness_high")
        wing = spot * self.config.float_("candidates.max_wing_width_pct")
        return self.gateway.chain(
            self.config.str_("underlying.symbol"), expiry, low - wing, high + wing
        )

    def _archive(self, chain, spot, now, expiry, cycle_id: str):
        """Write down the chain this decision was made from.

        Historical option quotes for a past 10:00 cannot be fetched back later:
        an expired contract's book is gone. If the snapshot is not recorded now
        then this session can never be labelled honestly, and the classifier
        loses a training row permanently.

        A day already on disk is not an error. Re-running a cycle must not
        rewrite evidence, and it must not stop the cycle either.
        """
        directory = self.config.path_("paths.chain_archive")
        try:
            return archive.write(
                archive.ChainSnapshot(
                    session_date=now.date(),
                    taken_at=now,
                    spot=spot,
                    expiry=expiry,
                    entries=list(chain),
                    cycle_id=cycle_id,
                ),
                directory,
            )
        except DataError:
            existing = archive.path_for(directory, now.date())
            if existing.exists():
                return existing
            raise

    def _probability(
        self, family: Family, snapshot: feature_engine.FeatureSet, variance_history: Sequence[float]
    ) -> tuple[float, str]:
        if self.decision_probability is not None:
            return (
                self.decision_probability,
                self.decision_source or "active deterministic strategy profile",
            )
        model = self.models.get(family)
        if model is not None:
            return model.probability(snapshot.vector(model.feature_names)), "classifier"
        regime = self.rule.regime(snapshot.values["iv_total"], variance_history)
        # The yardstick is named in the receipt because the same rule returns a
        # different regime depending on what it is compared against, and a
        # reader of the ledger cannot otherwise tell which comparison was made.
        return (
            self.rule.probability(family, regime),
            f"regime rule ({regime}, implied history of {len(variance_history)} readings)",
        )

    # -------------------------------------------------------------------- cycle

    def run_cycle(
        self,
        prior_returns: Sequence[float],
        variance_history: Sequence[float],
        family_pnl: dict[str, Sequence[float]],
    ) -> CycleResult:
        cycle_id = new_cycle_id()
        symbol = self.config.str_("underlying.symbol")

        now, market_open = self.gateway.clock()
        account = self.gateway.account()
        spot, _ = self.gateway.spot(symbol)
        session_close = self._session_close(now)

        live_positions = self.gateway.positions()
        if live_positions:
            # A second session must not pretend its portfolio tail starts at
            # zero. Until every existing option leg is attributed and closed,
            # no new structure can be sized honestly.
            reason = (
                f"account has {len(live_positions)} open broker position(s); "
                "new entries are blocked until the portfolio is flat"
            )
            self.ledger.append(
                Record(
                    action=Action.RISK_HALT,
                    cycle_id=cycle_id,
                    rationale=f"No cycle today: {reason}.",
                    reject_reason="existing_positions",
                )
            )
            return CycleResult(cycle_id, True, reason)

        context = GateContext(
            config=self.config,
            now_exchange=now,
            session_close=session_close,
            is_trading_day=self.gateway.is_trading_day(now.date()),
            market_open=market_open,
            equity=account.equity,
            last_equity=account.last_equity,
            # Options buying power is an explicit account constraint. Falling
            # back to stock buying power turns an unavailable option limit into
            # a made-up permission to trade.
            buying_power=account.options_buying_power,
            spot=spot,
            open_structures=0,
            es_in_use=0.0,
            cumulative_fees=self._fees_paid(),
            kill_switch_path=self.config.path_("paths.kill_switch"),
            submission_cutoff=self.submission_cutoff,
        )

        session = run_session_gates(context)
        if not session.passed:
            failure = session.first_failure
            self.ledger.append(
                Record(
                    action=Action.RISK_HALT,
                    cycle_id=cycle_id,
                    rationale=f"No cycle today: {failure.detail}.",
                    checks=session.as_dicts(),
                    reject_reason=failure.name,
                )
            )
            return CycleResult(cycle_id, True, failure.detail)

        expiries = self.gateway.expirations(symbol, now.date())
        if expiries[0] != now.date():
            reason = (
                f"the nearest listed {symbol} expiry is {expiries[0]}, not today; "
                "this project trades same-day expiries only"
            )
            self.ledger.append(
                Record(action=Action.STAND_DOWN, cycle_id=cycle_id, rationale=reason.capitalize())
            )
            return CycleResult(cycle_id, True, reason)

        chain = self._chain(spot, expiries[0])
        archived = self._archive(chain, spot, now, expiries[0], cycle_id)
        snapshot = feature_engine.build(
            chain,
            spot,
            now,
            session_close,
            prior_returns,
            {
                str(name): list(family_pnl.get(str(name), []))
                for name in self.config.list_("structures.enabled")
            },
            rate=self.config.float_("reconstruction.risk_free_rate"),
        )
        self.ledger.append(
            Record(
                action=Action.SNAPSHOT,
                cycle_id=cycle_id,
                rationale=(
                    f"{symbol} at {spot:.2f} with {len(chain)} priceable contracts on the "
                    f"{expiries[0]} expiry; implied skew {snapshot.values['implied_skew']:+.5f}."
                ),
                features=snapshot.as_dict(),
                extra={"chain_archive": str(archived) if archived else None},
            )
        )

        result = CycleResult(cycle_id, True, "no family cleared its checks")
        es_in_use = 0.0
        opened = 0

        for family, candidates in build_candidates(chain, self.config, spot).items():
            if self.candidate_filter is not None:
                candidates = [
                    candidate for candidate in candidates if self.candidate_filter(candidate)
                ]
            if not candidates:
                self.ledger.append(
                    Record(
                        action=Action.STAND_DOWN,
                        cycle_id=cycle_id,
                        structure=str(family),
                        rationale=(
                            f"{family} stood down: the active profile supplied no candidates "
                            "consistent with its observed signal."
                        ),
                        extra=dict(self.receipt_context),
                    )
                )
                continue
            probability, source = self._probability(family, snapshot, variance_history)
            # Priced one at a time rather than in a comprehension, because a
            # live chain is a set of prints from different moments and a few of
            # them cross. A structure built across a crossed pair shows a profit
            # at every expiry price, which is a stale quote and not free money,
            # and Law 5 refuses to compute a risk for it. That refusal is about
            # the one structure. Pricing the family inside a single expression
            # let it end the family, and on 31 August it ended the session.
            priced: list[PricedCandidate] = []
            unpriceable: list[str] = []
            for candidate in candidates:
                try:
                    estimate = evaluate(
                        candidate.legs,
                        self.scenarios,
                        self.cost_model,
                        spot,
                        1,
                        self.config.float_("risk.es_confidence"),
                    )
                except UndefinedRiskError as error:
                    unpriceable.append(f"{candidate.description}: {error}")
                    continue
                priced.append(PricedCandidate(candidate, estimate))

            if unpriceable:
                self.ledger.append(
                    Record(
                        action=Action.CANDIDATE_REJECTED,
                        cycle_id=cycle_id,
                        structure=str(family),
                        rationale=(
                            f"{len(unpriceable)} of {len(candidates)} {family} structures "
                            "carry quotes that do not describe a risk and were not priced. "
                            "First: " + unpriceable[0]
                        ),
                        reject_reason="undefined_risk",
                        extra={"unpriceable": unpriceable[:10]},
                    )
                )
            consumed = cost_consumed(priced)
            if consumed:
                worst = min(consumed, key=lambda item: item.estimate.net_edge)
                self.ledger.append(
                    Record(
                        action=Action.CANDIDATE_REJECTED,
                        cycle_id=cycle_id,
                        structure=str(family),
                        rationale=(
                            f"{len(consumed)} of {len(priced)} priced {family} structures "
                            "showed a gross profit that execution cost consumed entirely, "
                            "and were refused before ranking. Worst: "
                            f"{worst.candidate.description} at a gross edge of "
                            f"{worst.estimate.gross_edge:,.2f} against "
                            f"{worst.estimate.cost.total:,.2f} of cost across "
                            f"{worst.estimate.cost.leg_count} legs, leaving "
                            f"{worst.estimate.net_edge:,.2f}."
                        ),
                        reject_reason="net_of_cost",
                        extra={
                            "consumed": len(consumed),
                            "priced": len(priced),
                            # The page can only draw a record that carries one,
                            # so the refusal this project is built on could not
                            # be drawn at all while the others could.
                            "waterfall": worst.estimate.waterfall(),
                            "worst": {
                                "description": worst.candidate.description,
                                "gross_edge": round(worst.estimate.gross_edge, 2),
                                "cost": round(worst.estimate.cost.total, 2),
                                "net_edge": round(worst.estimate.net_edge, 2),
                                "legs": worst.estimate.cost.leg_count,
                            },
                        },
                    )
                )

            ordered = rank(
                priced, [str(name) for name in self.config.list_("structures.tie_break_order")]
            )
            if not ordered:
                continue

            # Walk the ranking rather than standing or falling on its head.
            # Every candidate here still has to clear every check on its own,
            # so this loosens nothing: it stops one wide wing on the top-ranked
            # structure from taking the whole family down with it when the
            # structure ranked behind it is tradeable and nearly as good. The
            # ranking is already net of cost, so the first one that clears is
            # the best tradeable structure in the family by construction.
            portfolio = PortfolioState(
                equity=context.equity,
                buying_power=context.buying_power,
                open_structures=context.open_structures + opened,
                es_in_use=es_in_use,
            )
            candidate_context = replace(
                context,
                probability=probability,
                es_in_use=es_in_use,
                open_structures=context.open_structures + opened,
            )

            best = None
            first = first_size = first_report = None
            unsizable: list[str] = []
            for attempt, priced_candidate in enumerate(ordered[: self.max_attempts]):
                try:
                    size = size_position(priced_candidate.estimate, portfolio, self.config)
                except UndefinedRiskError as error:
                    # A structure whose risk cannot be computed is not sized, by
                    # Law 5, and one that shows a profit at every expiry price is
                    # a crossed or stale quote rather than free money. Skipping it
                    # is not swallowing the error: it is recorded below, and the
                    # candidate is refused rather than the session abandoned. The
                    # live chain is a set of prints from different moments and
                    # always contains a few of these.
                    unsizable.append(f"{priced_candidate.candidate.description}: {error}")
                    continue
                report = run_candidate_gates(
                    candidate_context, priced_candidate.candidate, priced_candidate.estimate, size
                )
                if report.passed and size.trades:
                    best = priced_candidate
                    break
                # Only the best refusal is written down. Recording a rejection
                # for every structure the ranking walked past would bury the
                # decision in its own arithmetic, and the receipt that matters
                # is why the family's best structure could not be opened.
                if attempt == 0:
                    first_size, first_report, first = size, report, priced_candidate

            if unsizable:
                self.ledger.append(
                    Record(
                        action=Action.CANDIDATE_REJECTED,
                        cycle_id=cycle_id,
                        structure=str(family),
                        rationale=(
                            f"{len(unsizable)} of the {len(ordered[: self.max_attempts])} "
                            f"ranked {family} structures could not be sized because their "
                            "quotes do not describe a risk. First: " + unsizable[0]
                        ),
                        reject_reason="undefined_risk",
                        extra={"unsizable": unsizable[:10]},
                    )
                )

            if best is None:
                if first is None:
                    continue
                self._record_rejection(
                    cycle_id, family, first, first_size, first_report, probability, source, result
                )
                continue

            self._open(
                cycle_id, family, best, size, report, probability, source, result, candidate_context
            )
            es_in_use += best.estimate.expected_shortfall * size.contracts
            opened += 1
            result.stood_down = False
            result.reason = f"{opened} structure(s) opened"

        if result.stood_down:
            self.ledger.append(
                Record(
                    action=Action.STAND_DOWN,
                    cycle_id=cycle_id,
                    rationale=(
                        "Stood down. No family produced a candidate that cleared its checks "
                        "once execution cost was priced in."
                    ),
                    structure=str(Family.STAND_DOWN),
                )
            )
        return result

    # ------------------------------------------------------------------ actions

    def _record_rejection(
        self, cycle_id, family, best, size, report, probability, source, result
    ) -> None:
        failure = report.first_failure
        reason = failure.name if failure is not None else "size_is_zero"
        detail = (
            failure.detail
            if failure is not None
            else f"sizing returned no contracts, limited by {size.binding_constraint}"
        )
        text = rationale_layer.deterministic_text(
            best.candidate, best.estimate, size, report, probability
        )
        self.ledger.append(
            Record(
                action=Action.CANDIDATE_REJECTED,
                cycle_id=cycle_id,
                structure=str(family),
                rationale=text,
                probability=probability,
                legs=best.candidate.leg_dicts(),
                net_price=round(best.estimate.profile.net_entry_debit, 4),
                cost_breakdown=best.estimate.cost.as_dict(),
                max_loss=round(best.estimate.profile.max_loss, 2),
                es_contribution=round(best.estimate.expected_shortfall, 2),
                contracts=size.contracts,
                checks=report.as_dicts(),
                reject_reason=reason,
                extra={
                    "probability_source": source,
                    "waterfall": best.estimate.waterfall(),
                    "sizing": size.as_dict(),
                },
            )
        )
        result.rejections.append({"family": str(family), "reason": reason, "detail": detail})

    def _open(
        self, cycle_id, family, best, size, report, probability, source, result, context
    ) -> None:
        brief = rationale_layer.build_brief(
            best.candidate, best.estimate, size, report, probability, source
        )
        fallback = rationale_layer.deterministic_text(
            best.candidate, best.estimate, size, report, probability
        )
        rationale = rationale_layer.narrate(brief, fallback)

        # This is the entry limit, so it includes only entry friction. Max loss
        # below still uses the all-in profile, including the assignment reserve.
        limit_price = round(
            self.cost_model.executable_debit(best.candidate.legs, size.contracts), 2
        )
        client_order_id = f"convex-{cycle_id}-{family}"[:48]

        # The rationale is durable before the order exists, never after. On a
        # dry run the order never comes to exist, so the action says so: a
        # rehearsal that wrote order_submitted would put an order in the
        # evidence that was never sent, and the ledger is the evidence.
        self.ledger.append(
            Record(
                action=Action.DRY_RUN if self.dry_run else Action.ORDER_SUBMITTED,
                cycle_id=cycle_id,
                structure=str(family),
                rationale=rationale.text,
                probability=probability,
                legs=best.candidate.leg_dicts(),
                net_price=limit_price,
                cost_breakdown=best.estimate.cost.as_dict(),
                max_loss=round(best.estimate.profile.max_loss * size.contracts, 2),
                es_contribution=round(best.estimate.expected_shortfall * size.contracts, 2),
                contracts=size.contracts,
                checks=report.as_dicts(),
                extra={
                    "probability_source": source,
                    "waterfall": best.estimate.waterfall(),
                    "sizing": size.as_dict(),
                    "client_order_id": client_order_id,
                    **rationale.as_dict(),
                    **self.receipt_context,
                },
            )
        )

        if self.dry_run:
            result.stood_down = False
            result.orders.append(
                {"family": str(family), "order_id": f"not sent, {client_order_id}"}
            )
            return

        try:
            order = self.gateway.submit_structure(
                list(best.candidate.legs), size.contracts, limit_price, client_order_id
            )
            resolution = resolve_order(
                self.gateway,
                order,
                size.contracts,
                timeout_seconds=self.config.float_("execution.order_status_timeout_seconds"),
                poll_seconds=self.config.float_("execution.order_poll_seconds"),
            )
        except ConvexError as error:
            self.ledger.append(
                Record(
                    action=Action.ORDER_REJECTED,
                    cycle_id=cycle_id,
                    structure=str(family),
                    rationale=f"The entry could not be verified: {error}",
                    contracts=size.contracts,
                    reject_reason="entry_submission_failed",
                    extra={"client_order_id": client_order_id},
                )
            )
            raise

        outcome = outcome_fields(resolution.order, cancel_requested=resolution.cancel_requested)
        if not resolution.filled:
            can_retry = (
                self.reprice_ticks
                and resolution.terminal
                and filled_quantity(resolution.order) == 0
            )
            if can_retry:
                # The first cancellation is its own broker fact, even if a
                # later rung fills. Never let a successful retry erase it.
                self.ledger.append(
                    Record(
                        action=Action.ORDER_REJECTED,
                        cycle_id=cycle_id,
                        structure=str(family),
                        rationale=(
                            f"Alpaca reports the initial entry {resolution.order.status} with "
                            f"0/{size.contracts} contracts filled; fresh-quote retry is permitted."
                        ),
                        contracts=size.contracts,
                        reject_reason="entry_not_filled",
                        outcome=outcome,
                        extra={
                            **self.receipt_context,
                            "client_order_id": client_order_id,
                            "reprice_enabled": True,
                        },
                    )
                )
                if self._retry_entry(
                    cycle_id, family, best, size, probability, source, result, context
                ):
                    return
            else:
                pending = not resolution.terminal
                self.ledger.append(
                    Record(
                        action=Action.ORDER_PENDING if pending else Action.ORDER_REJECTED,
                        cycle_id=cycle_id,
                        structure=str(family),
                        rationale=(
                            f"Alpaca reports {resolution.order.status} with "
                            f"{resolution.order.filled_qty}/{size.contracts} contracts filled; "
                            "the entry is not recorded as an open position."
                        ),
                        contracts=size.contracts,
                        reject_reason="entry_pending" if pending else "entry_not_filled",
                        outcome=outcome,
                        extra={"client_order_id": client_order_id},
                    )
                )
            raise DataError(
                f"entry {order.id} was not fully filled: {resolution.order.status} "
                f"{resolution.order.filled_qty}/{size.contracts}"
            )

        self.ledger.append(
            Record(
                action=Action.ORDER_FILLED,
                cycle_id=cycle_id,
                structure=str(family),
                rationale=(
                    f"Alpaca verified the multi-leg order {resolution.order.id} as fully filled "
                    f"for {resolution.order.filled_qty} contracts."
                ),
                contracts=size.contracts,
                legs=best.candidate.leg_dicts(),
                net_price=limit_price,
                cost_breakdown=best.estimate.cost.as_dict(),
                max_loss=round(best.estimate.profile.max_loss * size.contracts, 2),
                es_contribution=round(best.estimate.expected_shortfall * size.contracts, 2),
                outcome=outcome,
                extra={"client_order_id": client_order_id},
            )
        )
        result.orders.append({"family": str(family), "order_id": str(resolution.order.id)})

    def _retry_entry(
        self, cycle_id, family, best, original_size, probability, source, result, context
    ) -> bool:
        """Retry a canceled entry only from fresh quotes and fresh risk arithmetic."""
        symbols = [leg.contract.symbol for leg in best.candidate.legs]
        for attempt, tick in enumerate(self.reprice_ticks, start=1):
            # A canceled order is not permission to trade on the old state.
            # Each rung reads a new account, clock, spot, and option quote set.
            account = self.gateway.account()
            now, market_open = self.gateway.clock()
            spot, _ = self.gateway.spot(self.config.str_("underlying.symbol"))
            retry_context = replace(
                context,
                now_exchange=now,
                market_open=market_open,
                equity=account.equity,
                last_equity=account.last_equity,
                buying_power=account.options_buying_power,
                spot=spot,
            )
            session = run_session_gates(retry_context)
            if not session.passed:
                failure = session.first_failure
                self.ledger.append(
                    Record(
                        action=Action.RISK_HALT,
                        cycle_id=cycle_id,
                        structure=str(family),
                        rationale=f"Reprice rung {attempt} refused: {failure.detail}.",
                        checks=session.as_dicts(),
                        reject_reason=failure.name,
                        extra={**self.receipt_context, "reprice_attempt": attempt},
                    )
                )
                return False
            quotes = self.gateway.option_quotes(symbols)
            legs = tuple(
                replace(leg, entry=replace(leg.entry, quote=quotes[leg.contract.symbol]))
                for leg in best.candidate.legs
            )
            candidate = replace(best.candidate, legs=legs)
            portfolio = PortfolioState(
                retry_context.equity,
                retry_context.buying_power,
                retry_context.open_structures,
                retry_context.es_in_use,
            )
            estimate = evaluate(
                legs,
                self.scenarios,
                self.cost_model,
                retry_context.spot,
                1,
                self.config.float_("risk.es_confidence"),
            )
            limit = round(
                self.cost_model.executable_debit(legs)
                + tick * self.config.float_("costs.tick_size"),
                2,
            )
            estimate = at_limit(
                estimate, legs, self.cost_model, limit, self.config.float_("risk.es_confidence")
            )
            try:
                size = size_position(estimate, portfolio, self.config)
            except UndefinedRiskError:
                continue
            gates = run_candidate_gates(retry_context, candidate, estimate, size)
            if not gates.passed or not size.trades:
                self.ledger.append(
                    Record(
                        action=Action.CANDIDATE_REJECTED,
                        cycle_id=cycle_id,
                        structure=str(family),
                        rationale=f"Reprice rung {attempt} at {limit:.2f} refused: "
                        f"{gates.first_failure.detail if gates.first_failure else 'size is zero'}",
                        reject_reason=(
                            gates.first_failure.name if gates.first_failure else 'size_is_zero'
                        ),
                        extra={
                            **self.receipt_context,
                            "reprice_attempt": attempt,
                            "limit_price": limit,
                        },
                    )
                )
                continue
            client_id = f"convex-{cycle_id}-{family}-r{attempt}"[:48]
            self.ledger.append(
                Record(
                    action=Action.ORDER_SUBMITTED,
                    cycle_id=cycle_id,
                    structure=str(family),
                    rationale=f"Fresh-quote reprice rung {attempt}: submitting "
                    f"{size.contracts} lots at {limit:.2f} after all gates passed.",
                    contracts=size.contracts,
                    net_price=limit,
                    checks=gates.as_dicts(),
                    extra={
                        **self.receipt_context,
                        "client_order_id": client_id,
                        "reprice_attempt": attempt,
                        "waterfall": estimate.waterfall(),
                    },
                )
            )
            order = self.gateway.submit_structure(list(legs), size.contracts, limit, client_id)
            resolution = resolve_order(
                self.gateway,
                order,
                size.contracts,
                timeout_seconds=self.config.float_("execution.order_status_timeout_seconds"),
                poll_seconds=self.config.float_("execution.order_poll_seconds"),
            )
            outcome = outcome_fields(resolution.order, cancel_requested=resolution.cancel_requested)
            if resolution.filled:
                self.ledger.append(
                    Record(
                        action=Action.ORDER_FILLED,
                        cycle_id=cycle_id,
                        structure=str(family),
                        rationale=f"Alpaca verified reprice rung {attempt} as fully filled "
                        f"for {resolution.order.filled_qty} contracts.",
                        contracts=size.contracts,
                        legs=candidate.leg_dicts(),
                        net_price=limit,
                        cost_breakdown=estimate.cost.as_dict(),
                        max_loss=round(estimate.profile.max_loss * size.contracts, 2),
                        es_contribution=round(estimate.expected_shortfall * size.contracts, 2),
                        outcome=outcome,
                        extra={
                            **self.receipt_context,
                            "client_order_id": client_id,
                            "reprice_attempt": attempt,
                        },
                    )
                )
                result.orders.append({"family": str(family), "order_id": str(resolution.order.id)})
                return True
            self.ledger.append(
                Record(
                    action=Action.ORDER_REJECTED,
                    cycle_id=cycle_id,
                    structure=str(family),
                    rationale=f"Alpaca reports reprice rung {attempt} {resolution.order.status} "
                    f"with {resolution.order.filled_qty}/{size.contracts} filled.",
                    contracts=size.contracts,
                    reject_reason="entry_not_filled",
                    outcome=outcome,
                    extra={
                        **self.receipt_context,
                        "client_order_id": client_id,
                        "reprice_attempt": attempt,
                    },
                )
            )
        return False


    def _fees_paid(self) -> float:
        """Actual settled fees plus a conservative reserve for open verified entries.

        A submission is not a trade and must never consume the fee budget. A
        broker-reconciled close carries actual fees; a verified entry not yet
        closed carries only its configured entry-fee reserve until Alpaca has
        published the day-level activity receipt.
        """
        settled_entry_ids: set[str] = set()
        actual = 0.0
        rows = list(self.ledger.read())
        for record in rows:
            if record.get("action") != Action.POSITION_RECONCILED.value:
                continue
            outcome = record.get("outcome") or {}
            entry_order_id = str(outcome.get("entry_order_id", ""))
            if not entry_order_id or "broker_fees" not in outcome:
                raise DataError(
                    f"reconciled position {record.get('seq')} is missing its broker fee receipt"
                )
            settled_entry_ids.add(entry_order_id)
            actual += float(outcome["broker_fees"])

        reserved = 0.0
        for record in rows:
            if record.get("action") not in {
                Action.ORDER_FILLED.value,
                Action.ORDER_RECONCILED.value,
            }:
                continue
            outcome = record.get("outcome") or {}
            order_id = str(outcome.get("order_id", ""))
            if order_id in settled_entry_ids:
                continue
            if str(outcome.get("status", "")).lower() != "filled":
                continue
            breakdown = record.get("cost_breakdown")
            contracts = record.get("contracts")
            if breakdown is None or contracts is None or "fees" not in breakdown:
                raise DataError(
                    f"verified entry {order_id} is missing its fee reserve receipt"
                )
            reserved += float(breakdown["fees"]) * int(contracts)
        return actual + reserved
