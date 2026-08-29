"""The written rationale that precedes every action.

Law 6: the agent explains before it acts, and the explanation is persisted
before the order is sent. An agent that trades without explaining is a script
with an API key.

The hard rule of this module, and the reason it is small: the language model
may narrate and it may explain, but it may never compute. Every number in every
sentence below is computed by the deterministic core and interpolated in. The
model is handed a finished brief of measured figures and asked to put them into
prose. If it is unavailable, unconfigured, or slow, the deterministic brief is
what gets written to the ledger and shown on the dashboard, and the record says
which one it was. Nothing waits on it and nothing depends on it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from convex.edge import EdgeEstimate
from convex.gates import GateReport
from convex.sizing import SizeDecision
from convex.structures.base import Candidate

FEATHERLESS_ENDPOINT = "https://api.featherless.ai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are the narration layer of an options trading agent. You will be given "
    "a brief of figures that have already been computed. Restate it as two or "
    "three plain sentences a trader would find useful. You must not compute, "
    "estimate, round differently, or introduce any number that is not in the "
    "brief, and you must not add advice, caveats or disclaimers."
)


@dataclass(frozen=True)
class Rationale:
    """The text written to the ledger, and where it came from."""

    text: str
    source: str  # "deterministic" or "featherless"
    brief: str

    def as_dict(self) -> dict[str, str]:
        return {"rationale_source": self.source, "brief": self.brief}


def build_brief(
    candidate: Candidate,
    estimate: EdgeEstimate,
    size: SizeDecision,
    report: GateReport,
    probability: float,
    probability_source: str,
) -> str:
    """A factual brief. Every figure here was measured or computed upstream."""
    waterfall = estimate.waterfall()
    lines = [
        f"structure: {candidate.description}",
        f"family: {candidate.family}",
        f"probability of a profitable outcome: {probability:.3f} ({probability_source})",
        f"gross expected result across {len(estimate.net_outcomes)} historical sessions: "
        f"{waterfall['gross_edge']:.2f} dollars",
        f"execution cost: {estimate.cost.total:.2f} dollars across "
        f"{estimate.cost.leg_count} legs, of which {estimate.cost.half_spread:.2f} is "
        f"half-spread and {estimate.cost.exit_reserve:.2f} is the reserve to close the shorts",
        f"net expected result: {estimate.net_edge:.2f} dollars",
        f"win rate across those sessions: {estimate.win_rate:.1%}",
        f"worst case: {estimate.profile.max_loss:.2f} dollars per lot at an expiry price of "
        f"{estimate.profile.max_loss_price:g}",
        f"one percent tail: {estimate.expected_shortfall:.2f} dollars per lot",
        f"entered for a {'credit' if estimate.profile.is_credit else 'debit'} of "
        f"{abs(estimate.profile.net_entry_debit):.2f} per share",
        f"breakevens: {', '.join(f'{level:g}' for level in estimate.profile.breakevens) or 'none'}",
        f"size: {size.contracts} contracts, limited by {size.binding_constraint}",
        "checks: " + "; ".join(
            f"{result.name} {'passed' if result.passed else 'FAILED'}" for result in report.results
        ),
    ]
    return "\n".join(lines)


def deterministic_text(
    candidate: Candidate,
    estimate: EdgeEstimate,
    size: SizeDecision,
    report: GateReport,
    probability: float,
) -> str:
    """The rationale when nothing narrates it: still complete, just plainer."""
    failure = report.first_failure
    if failure is not None:
        return (
            f"Refused {candidate.description}. {failure.detail.capitalize()}. "
            f"Gross expected result was {estimate.gross_edge:.2f} dollars against "
            f"{estimate.cost.total:.2f} dollars of execution cost across "
            f"{estimate.cost.leg_count} legs, leaving {estimate.net_edge:.2f} net."
        )
    return (
        f"Entering {size.contracts} lots of {candidate.description} at a probability of "
        f"{probability:.3f}. Gross expected result {estimate.gross_edge:.2f} dollars less "
        f"{estimate.cost.total:.2f} dollars of cost leaves {estimate.net_edge:.2f} net, with a "
        f"{estimate.win_rate:.0%} win rate across the scenario set. Worst case is "
        f"{estimate.profile.max_loss:.2f} dollars per lot at {estimate.profile.max_loss_price:g} "
        f"and the one percent tail is {estimate.expected_shortfall:.2f}; size is limited by "
        f"{size.binding_constraint}."
    )


def narrate(brief: str, fallback: str, timeout_seconds: float = 8.0) -> Rationale:
    """Ask Featherless to phrase the brief, and never depend on the answer."""
    api_key = os.environ.get("FEATHERLESS_API_KEY", "").strip()
    model = os.environ.get("FEATHERLESS_MODEL", "").strip()
    if not api_key or not model:
        return Rationale(text=fallback, source="deterministic", brief=brief)

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": brief},
            ],
            "temperature": 0.2,
            "max_tokens": 220,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        FEATHERLESS_ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError) as exc:
        # The narration is decoration on a decision that is already made and
        # already justified. Its failure is recorded and the deterministic
        # rationale stands; it never blocks or alters an order.
        return Rationale(
            text=fallback,
            source=f"deterministic (featherless unavailable: {type(exc).__name__})",
            brief=brief,
        )
    if not text:
        return Rationale(text=fallback, source="deterministic (featherless returned nothing)", brief=brief)
    return Rationale(text=text, source="featherless", brief=brief)
