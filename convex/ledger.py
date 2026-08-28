"""The decision ledger.

Law 4: every decision leaves a receipt, including the decisions not to trade.
The ledger is append-only JSONL. It is the demo, the write-up evidence, and the
source of every number on the dashboard, so it is written before anything that
could produce a record exists.

Nothing in this module ever rewrites or truncates a line. Records are appended
with an fsync so that a decision survives a process that dies immediately after
making it, which is the case that matters: an order can reach Alpaca and the
process can die before the receipt is durable.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator

from convex.errors import ConvexError


class Action(StrEnum):
    """What the agent did, or declined to do, at this point in a cycle."""

    CANDIDATE_PRICED = "candidate_priced"
    CANDIDATE_REJECTED = "candidate_rejected"
    STAND_DOWN = "stand_down"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    POSITION_CLOSED = "position_closed"
    RISK_HALT = "risk_halt"
    CALIBRATION = "calibration"
    SNAPSHOT = "snapshot"


@dataclass(frozen=True)
class Record:
    """One immutable ledger line.

    The optional fields are the ones the spec names as required evidence. They
    are left as None when they genuinely do not apply to the action rather than
    being filled with a placeholder, so a reader can distinguish "no position
    was sized" from "size was zero".
    """

    action: Action
    cycle_id: str
    rationale: str
    structure: str | None = None
    probability: float | None = None
    features: dict[str, float] | None = None
    legs: list[dict[str, Any]] | None = None
    net_price: float | None = None
    cost_breakdown: dict[str, float] | None = None
    max_loss: float | None = None
    es_contribution: float | None = None
    contracts: int | None = None
    checks: list[dict[str, Any]] | None = None
    reject_reason: str | None = None
    outcome: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self, timestamp: datetime, sequence: int) -> str:
        payload: dict[str, Any] = {
            "ts": timestamp.isoformat(),
            "seq": sequence,
            "cycle_id": self.cycle_id,
            "action": str(self.action),
            "rationale": self.rationale,
        }
        for key in (
            "structure",
            "probability",
            "features",
            "legs",
            "net_price",
            "cost_breakdown",
            "max_loss",
            "es_contribution",
            "contracts",
            "checks",
            "reject_reason",
            "outcome",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        payload.update(self.extra)
        return json.dumps(payload, separators=(",", ":"), sort_keys=False, default=_encode)


def _encode(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return str(value)
    raise TypeError(f"ledger cannot serialise {type(value).__name__}: {value!r}")


class Ledger:
    """An append-only JSONL sink."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sequence = self._count_existing()

    def _count_existing(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def append(self, record: Record) -> str:
        """Write one record durably and return the line as written."""
        timestamp = datetime.now(timezone.utc)
        with self._lock:
            self._sequence += 1
            line = record.to_json(timestamp, self._sequence)
            if "\n" in line:
                raise ConvexError("a ledger record serialised to more than one line")
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return line

    def read(self) -> Iterator[dict[str, Any]]:
        """Yield every record written so far, oldest first."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ConvexError(f"{self.path}:{number} is not valid JSON: {exc}") from exc


def new_cycle_id() -> str:
    """Identifier tying every record produced by one decision cycle together."""
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
