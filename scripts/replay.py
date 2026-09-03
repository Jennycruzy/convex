"""Re-price an archived session and show what execution cost took away.

Law 7 says cost is a first-class citizen, and the ledger did not show it. The
ranking is ordered by net edge and the walk only ever reaches its first few
entries, so a structure whose whole gross profit goes to the cost of crossing
is demoted out of contention before any check sees it. The net-of-cost check
can therefore only ever fire on a candidate cost has already vindicated, and a
reader of the ledger would conclude cost never bit once.

This runs the same pricing over an archived chain and counts them, so the claim
can be checked rather than taken on trust. It reads only files in this
repository, reaches no network, sends no order and writes nothing to the
ledger.

Read the header before the table. A recorded chain is evidence and is never
rewritten, so the archive holds the first snapshot taken in a session, which is
not always the one a particular ledger entry was decided on. The 1 September
archive is the 09:37 snapshot from a run that failed before the entry, not the
10:12 cycle that stood the agent down. The scenario set is paired the same way,
as the newest that was not built after the chain was taken, because pricing a
morning against a distribution built later in the day would leak the afternoon
into it. So this measures what cost does to a real SPY 0DTE book, which is the
claim being made, and it does not reproduce any single recorded decision.

It prices every family that has a builder, not the families
``structures.enabled`` currently permits. Those are separate questions. The
permission list is a live risk decision and is empty at the moment, because the
2 September audit withdrew every family from trading; reading it here would
mean a stand-down erased the measurement too, and the table would go blank at
exactly the moment the claim most needs checking. Cost destroys a candidate
whether or not the agent is allowed to open it. Pass --families to narrow it.

Run it with:  .venv/bin/python -m scripts.replay
              .venv/bin/python -m scripts.replay --session 2026-09-01
              .venv/bin/python -m scripts.replay --families put_bwb,straddle
              .venv/bin/python -m scripts.replay --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from convex import archive, scenarios
from convex.agent import cost_consumed
from convex.config import load
from convex.costs import CostModel
from convex.edge import evaluate
from convex.errors import ConvexError, DataError, UndefinedRiskError
from convex.structures.base import Candidate, Family
from convex.structures.builders import BUILDABLE_FAMILIES, build_candidates


def pair_scenarios(directory: Path, chain_taken_at, config) -> Path:
    """The scenario set the session was priced against.

    The archives are stamped with the moment they were built, so the one that
    belongs to a chain is the newest that is not from after it. Refusing when
    none qualifies matters more than it looks: pricing a session against a
    distribution built later in the day would quietly leak the afternoon into
    a decision that was made in the morning.
    """
    available = scenarios.archived(directory)
    if not available:
        raise DataError(f"no scenario archives in {directory}")
    naive = chain_taken_at.replace(tzinfo=None)
    earlier = [path for path in available if scenarios.load(path, config).built_at <= naive]
    if not earlier:
        raise DataError(
            f"every scenario archive in {directory} was built after the chain was taken "
            f"at {chain_taken_at.isoformat()}, so none of them priced it"
        )
    return earlier[-1]


def replay(session: date, config, families=BUILDABLE_FAMILIES) -> dict:
    chain_path = config.path_("paths.chain_archive") / f"chain-{session.isoformat()}.json.gz"
    if not chain_path.exists():
        raise DataError(f"no archived chain for {session.isoformat()} at {chain_path}")
    snapshot = archive.read(chain_path)
    scenario_path = pair_scenarios(config.path_("paths.scenario_archive"), snapshot.taken_at, config)
    scenario_set = scenarios.load(scenario_path, config)

    cost_model = CostModel(
        slippage_ticks_per_leg=config.float_("costs.slippage_ticks_per_leg"),
        tick_size=config.float_("costs.tick_size"),
        per_contract_fee=config.float_("costs.per_contract_fee"),
        regulatory_fee_per_contract=config.float_("costs.regulatory_fee_per_contract"),
        exit_reserve_legs=config.str_("costs.exit_reserve_legs"),
    )
    confidence = config.float_("risk.es_confidence")
    built = build_candidates(snapshot.entries, config, snapshot.spot, families)

    report: dict = {
        "session": session.isoformat(),
        "chain": str(chain_path),
        "scenarios": str(scenario_path),
        "spot": snapshot.spot,
        "expiry": snapshot.expiry.isoformat(),
        "priced_families": [str(family) for family in families],
        "min_leg_premium": config.float_("candidates.min_leg_premium"),
        "families": {},
    }
    for family, candidates in built.items():
        priced = []
        unpriceable = 0
        for candidate in candidates:
            try:
                estimate = evaluate(
                    candidate.legs, scenario_set, cost_model, snapshot.spot, 1, confidence
                )
            except UndefinedRiskError:
                # A crossed or stale print is not a candidate. It is counted and
                # skipped, never silently dropped.
                unpriceable += 1
                continue
            priced.append(_Priced(candidate, estimate))
        consumed = cost_consumed(priced)
        entry: dict = {
            "priced": len(priced),
            "unpriceable": unpriceable,
            "consumed": len(consumed),
        }
        if consumed:
            worst = min(consumed, key=lambda item: item.estimate.net_edge)
            entry["worst"] = {
                "description": worst.candidate.description,
                "gross_edge": round(worst.estimate.gross_edge, 2),
                "cost": round(worst.estimate.cost.total, 2),
                "net_edge": round(worst.estimate.net_edge, 2),
                "legs": worst.estimate.cost.leg_count,
            }
        report["families"][str(family)] = entry
    return report


class _Priced:
    """The shape cost_consumed reads, kept local so replay owns no policy."""

    __slots__ = ("candidate", "estimate")

    def __init__(self, candidate: Candidate, estimate) -> None:
        self.candidate = candidate
        self.estimate = estimate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        help="the archived session to replay, as YYYY-MM-DD. Defaults to the most recent.",
    )
    parser.add_argument(
        "--families",
        help=(
            "comma-separated families to price. Defaults to every family that has a "
            "builder, which is deliberately not the structures.enabled permission list."
        ),
    )
    parser.add_argument("--json", help="write the full report here as well as printing it")
    arguments = parser.parse_args()

    config = load()
    if arguments.families:
        names = [name.strip() for name in arguments.families.split(",") if name.strip()]
        try:
            families = tuple(Family(name) for name in names)
        except ValueError as error:
            raise DataError(
                f"--families names something that is not a family: {error}. "
                f"Known families are {', '.join(str(f) for f in BUILDABLE_FAMILIES)}."
            ) from None
        unbuildable = [f for f in families if f not in BUILDABLE_FAMILIES]
        if unbuildable:
            raise DataError(
                f"--families names {', '.join(str(f) for f in unbuildable)}, which has no builder"
            )
    else:
        families = BUILDABLE_FAMILIES
    if arguments.session:
        session = date.fromisoformat(arguments.session)
    else:
        recorded = archive.sessions(config.path_("paths.chain_archive"))
        if not recorded:
            raise DataError(
                f"no archived chains in {config.path_('paths.chain_archive')}, so there "
                "is nothing to replay"
            )
        session = recorded[-1]

    report = replay(session, config, families)
    print(f"session {report['session']}, spot {report['spot']}, expiry {report['expiry']}")
    print(f"chain     {report['chain']}")
    print(f"scenarios {report['scenarios']}")
    print(
        f"pricing   {', '.join(report['priced_families'])} "
        f"at a {report['min_leg_premium']:g} leg premium floor"
    )
    print()
    print(f"{'family':16}{'priced':>8}{'consumed':>10}{'share':>8}   worst gross to net")
    total_priced = total_consumed = 0
    for name, entry in report["families"].items():
        total_priced += entry["priced"]
        total_consumed += entry["consumed"]
        share = f"{entry['consumed'] / entry['priced']:7.1%}" if entry["priced"] else "    n/a"
        worst = entry.get("worst")
        detail = (
            f"{worst['gross_edge']:8.2f} to {worst['net_edge']:9.2f} over {worst['legs']} legs"
            if worst
            else "cost took nothing"
        )
        print(f"{name:16}{entry['priced']:>8}{entry['consumed']:>10}{share}   {detail}")
    if total_priced:
        print()
        print(
            f"{total_consumed} of {total_priced} priced candidates showed a gross profit "
            f"that execution cost consumed entirely ({total_consumed / total_priced:.1%})."
        )
    report["totals"] = {"priced": total_priced, "consumed": total_consumed}
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(report, indent=2))
        print(f"written to {arguments.json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
