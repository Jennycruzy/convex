"""Recorded chains, so the classifier has something honest to learn from.

Nothing in this project trains on simulated data, and the one exception the
rules allow is training on *recorded* real chains. This is the recorder. Every
cycle writes the 10:00 snapshot it actually saw, with every contract, its
quote, its Greeks and its open interest, to a dated file, and training reads those files
back rather than re-fetching a chain that no longer exists.

That is not a convenience. Historical option quotes for a past 10:00 are not
something the market data API will hand back later: an expired contract's book
is gone. If the snapshot the decision was made on is not written down at the
time, the label for that day can never be reconstructed honestly, and a model
trained on prices that were not the prices is worse than no model.

The archive is also what makes a decision auditable. A judge can point at a
refusal, open the chain it was made from, and recompute it.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Sequence

from convex.errors import DataError
from convex.instruments import ChainEntry, Greeks, OptionContract, Quote, Right

FORMAT_VERSION = 1


def _entry_to_dict(entry: ChainEntry) -> dict:
    contract = entry.contract
    greeks = entry.greeks
    return {
        "symbol": contract.symbol,
        "underlying": contract.underlying,
        "right": str(contract.right),
        "strike": contract.strike,
        "expiry": contract.expiry.isoformat(),
        "multiplier": contract.multiplier,
        "bid": entry.quote.bid,
        "ask": entry.quote.ask,
        "bid_size": entry.quote.bid_size,
        "ask_size": entry.quote.ask_size,
        "quoted_at": entry.quote.timestamp.isoformat(),
        "open_interest": entry.open_interest,
        "volume": entry.volume,
        "greeks": (
            None
            if greeks is None
            else {
                "delta": greeks.delta,
                "gamma": greeks.gamma,
                "theta": greeks.theta,
                "vega": greeks.vega,
                "rho": greeks.rho,
                "implied_volatility": greeks.implied_volatility,
            }
        ),
    }


def _entry_from_dict(row: dict) -> ChainEntry:
    greeks = row.get("greeks")
    return ChainEntry(
        contract=OptionContract(
            symbol=row["symbol"],
            underlying=row["underlying"],
            right=Right.CALL if row["right"] == str(Right.CALL) else Right.PUT,
            strike=float(row["strike"]),
            expiry=date.fromisoformat(row["expiry"]),
            multiplier=int(row["multiplier"]),
        ),
        quote=Quote(
            symbol=row["symbol"],
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            bid_size=int(row["bid_size"]),
            ask_size=int(row["ask_size"]),
            timestamp=datetime.fromisoformat(row["quoted_at"]),
        ),
        greeks=(
            None
            if greeks is None
            else Greeks(
                delta=float(greeks["delta"]),
                gamma=float(greeks["gamma"]),
                theta=float(greeks["theta"]),
                vega=float(greeks["vega"]),
                rho=float(greeks["rho"]),
                implied_volatility=float(greeks["implied_volatility"]),
            )
        ),
        open_interest=row.get("open_interest"),
        volume=row.get("volume"),
    )


@dataclass(frozen=True)
class ChainSnapshot:
    """One recorded 10:00 chain, and the state of the world around it."""

    session_date: date
    taken_at: datetime
    spot: float
    expiry: date
    entries: list[ChainEntry]
    cycle_id: str | None = None

    def __post_init__(self) -> None:
        if not self.entries:
            raise DataError(f"the {self.session_date} snapshot has no contracts in it")
        if self.spot <= 0.0:
            raise DataError(f"the {self.session_date} snapshot records a spot of {self.spot}")


def path_for(directory: Path, session_date: date) -> Path:
    return directory / f"chain-{session_date.isoformat()}.json.gz"


def write(snapshot: ChainSnapshot, directory: Path) -> Path:
    """Record one snapshot. An existing day is never silently overwritten."""
    directory.mkdir(parents=True, exist_ok=True)
    path = path_for(directory, snapshot.session_date)
    if path.exists():
        raise DataError(
            f"{path} already exists; a recorded chain is evidence and is not rewritten"
        )
    payload = {
        "format": FORMAT_VERSION,
        "session_date": snapshot.session_date.isoformat(),
        "taken_at": snapshot.taken_at.isoformat(),
        "spot": snapshot.spot,
        "expiry": snapshot.expiry.isoformat(),
        "cycle_id": snapshot.cycle_id,
        "entries": [_entry_to_dict(entry) for entry in snapshot.entries],
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return path


def read(path: Path) -> ChainSnapshot:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    version = payload.get("format")
    if version != FORMAT_VERSION:
        raise DataError(
            f"{path} is archive format {version}, this build reads {FORMAT_VERSION}"
        )
    return ChainSnapshot(
        session_date=date.fromisoformat(payload["session_date"]),
        taken_at=datetime.fromisoformat(payload["taken_at"]),
        spot=float(payload["spot"]),
        expiry=date.fromisoformat(payload["expiry"]),
        entries=[_entry_from_dict(row) for row in payload["entries"]],
        cycle_id=payload.get("cycle_id"),
    )


def sessions(directory: Path) -> list[date]:
    """Which sessions have a recorded chain, oldest first."""
    if not directory.is_dir():
        return []
    days: list[date] = []
    for path in directory.glob("chain-*.json.gz"):
        try:
            days.append(date.fromisoformat(path.stem.removeprefix("chain-").removesuffix(".json")))
        except ValueError as error:
            raise DataError(f"{path} is not a recognisable archive filename") from error
    return sorted(days)


def read_all(directory: Path) -> Iterator[ChainSnapshot]:
    """Every recorded snapshot, oldest first."""
    for day in sessions(directory):
        yield read(path_for(directory, day))
