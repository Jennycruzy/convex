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
from typing import Sequence
from zoneinfo import ZoneInfo

from convex import archive
from convex import features as feature_engine
from convex import rationale as rationale_layer
from convex.classifier import RegimeRule, StructureModel
from convex.config import Config
from convex.costs import CostModel
from convex.data.alpaca import AlpacaGateway
from convex.edge import EdgeEstimate, evaluate
from convex.errors import ConvexError, DataError
from convex.gates import GateContext, GateReport, run_candidate_gates, run_session_gates
from convex.instruments import ChainEntry
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
    ) -> None:
        self.gateway = gateway
        self.config = config
        self.ledger = ledger
        self.scenarios = scenarios
        self.models = models or {}
        self.cost_model = CostModel.from_config(config)
        self.rule = RegimeRule()
        self.submission_cutoff = submission_cutoff
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
        model = self.models.get(family)
        if model is not None:
            return model.probability(snapshot.vector(model.feature_names)), "classifier"
        regime = self.rule.regime(snapshot.values["iv_total"], variance_history)
        return self.rule.probability(family, regime), f"regime rule ({regime})"

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

        context = GateContext(
            config=self.config,
            now_exchange=now,
            session_close=session_close,
            is_trading_day=self.gateway.is_trading_day(now.date()),
            market_open=market_open,
            equity=account.equity,
            last_equity=account.last_equity,
            buying_power=account.options_buying_power or account.buying_power,
            spot=spot,
            open_structures=len(self.gateway.positions()),
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
            chain, spot, now, session_close, prior_returns, family_pnl
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
            probability, source = self._probability(family, snapshot, variance_history)
            priced = [
                PricedCandidate(
                    candidate,
                    evaluate(
                        candidate.legs,
                        self.scenarios,
                        self.cost_model,
                        spot,
                        1,
                        self.config.float_("risk.es_confidence"),
                    ),
                )
                for candidate in candidates
            ]
            ordered = rank(priced, [str(name) for name in self.config.list_("structures.tie_break_order")])
            if not ordered:
                continue

            best = ordered[0]
            portfolio = PortfolioState(
                equity=context.equity,
                buying_power=context.buying_power,
                open_structures=context.open_structures + opened,
                es_in_use=es_in_use,
            )
            size = size_position(best.estimate, portfolio, self.config)
            candidate_context = replace(
                context,
                probability=probability,
                es_in_use=es_in_use,
                open_structures=context.open_structures + opened,
            )
            report = run_candidate_gates(candidate_context, best.candidate, best.estimate, size)

            if not report.passed or not size.trades:
                self._record_rejection(
                    cycle_id, family, best, size, report, probability, source, result
                )
                continue

            self._open(
                cycle_id, family, best, size, report, probability, source, result
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

    def _open(self, cycle_id, family, best, size, report, probability, source, result) -> None:
        brief = rationale_layer.build_brief(
            best.candidate, best.estimate, size, report, probability, source
        )
        fallback = rationale_layer.deterministic_text(
            best.candidate, best.estimate, size, report, probability
        )
        rationale = rationale_layer.narrate(brief, fallback)

        limit_price = round(best.estimate.profile.net_entry_debit, 2)
        client_order_id = f"convex-{cycle_id}-{family}"[:48]

        # The rationale is durable before the order exists, never after.
        self.ledger.append(
            Record(
                action=Action.ORDER_SUBMITTED,
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
                },
            )
        )

        order = self.gateway.submit_structure(
            list(best.candidate.legs), size.contracts, limit_price, client_order_id
        )
        self.ledger.append(
            Record(
                action=Action.ORDER_FILLED,
                cycle_id=cycle_id,
                structure=str(family),
                rationale=f"Alpaca accepted the multi-leg order as {order.id}.",
                contracts=size.contracts,
                outcome={
                    "order_id": str(order.id),
                    "status": str(order.status),
                    "submitted_at": str(order.submitted_at),
                },
            )
        )
        result.orders.append({"family": str(family), "order_id": str(order.id)})

    def _fees_paid(self) -> float:
        """Execution cost paid so far, read back out of the ledger."""
        total = 0.0
        for record in self.ledger.read():
            if record.get("action") == Action.ORDER_FILLED.value:
                continue
            breakdown = record.get("cost_breakdown")
            if record.get("action") == Action.ORDER_SUBMITTED.value and breakdown:
                total += float(breakdown.get("total", 0.0))
        return total
