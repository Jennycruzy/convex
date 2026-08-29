"""Writing a measurement back into the configuration.

The transform is a pure function over the file's text, which is the point: it
can be checked exhaustively here without a market, and the script that calls it
does nothing but decide whether it is allowed to run.

What is being defended is narrow. The file's comments carry the MEASURED and
BOUND markings that the provenance lists are checked against, so a rewrite that
loses them breaks a different test in a way that is hard to trace back here.
And a measurement is two edits that have to land together or not at all.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from convex.config import load
from convex.errors import ConfigError
from convex.measured import apply_measurement, write_atomically

WHEN = date(2026, 8, 31)
BY = "scripts/calibrate_costs.py"
KEY = "liquidity.max_relative_spread"


@pytest.fixture
def original() -> str:
    return (Path(__file__).resolve().parent.parent / "config" / "convex.yaml").read_text()


def measure(text: str, value: float = 0.0731, key: str = KEY) -> str:
    return apply_measurement(text, key, value, WHEN, BY)


def test_the_value_is_replaced_and_marked_with_what_measured_it(original):
    updated = measure(original)
    line = next(
        line for line in updated.splitlines()
        if line.strip().startswith("max_relative_spread:")
    )
    assert "0.0731" in line
    assert f"MEASURED {WHEN.isoformat()} by {BY}" in line


def test_the_key_stops_blocking_the_session(original, tmp_path):
    updated = measure(original)
    path = tmp_path / "convex.yaml"
    path.write_text(updated)
    config = load(path)
    assert KEY not in config.hypotheses()
    assert config.float_(KEY) == pytest.approx(0.0731)


def test_nothing_else_leaves_the_blocking_list(original, tmp_path):
    """Exactly one key clears, and the bounds are untouched."""
    before = load(
        _write(tmp_path / "before.yaml", original)
    )
    after = load(_write(tmp_path / "after.yaml", measure(original)))
    assert set(before.hypotheses()) - set(after.hypotheses()) == {KEY}
    assert before.bounds() == after.bounds()


def test_every_comment_survives(original):
    """They are load-bearing: another test reads the markings out of them."""
    updated = measure(original)
    kept = [line.strip() for line in updated.splitlines() if line.strip().startswith("#")]
    was = [line.strip() for line in original.splitlines() if line.strip().startswith("#")]
    assert kept == was


def test_only_the_two_intended_lines_change(original):
    """A rewrite that touched anything else would be very hard to notice."""
    before = original.splitlines()
    after = measure(original).splitlines()
    assert len(before) - len(after) == 1
    changed = [line for line in after if line not in before]
    assert len(changed) == 1
    assert changed[0].strip().startswith("max_relative_spread:")


def test_a_key_in_a_section_that_does_not_have_it_is_refused(original):
    with pytest.raises(ConfigError, match="was not found"):
        measure(original, key="costs.max_relative_spread")


def test_a_key_already_measured_is_refused_rather_than_written_twice(original):
    """The second call has no blocking entry to clear, so it must not proceed:
    writing the value while leaving the lists alone is the half-done state."""
    once = measure(original)
    with pytest.raises(ConfigError, match="not listed under provenance.hypothesis"):
        measure(once)


def test_a_closed_market_is_refused_and_the_file_is_left_byte_identical(tmp_path, original):
    """The refusal that matters most.

    A relative spread read off a shut market is the width of a book nobody is
    quoting. Saturday's chain gave a median of 13.9% that way, which would have
    been written in and then trusted as a measurement.
    """
    from scripts.calibrate_costs import _apply

    path = _write(tmp_path / "convex.yaml", original)
    config = load(path)
    before = path.read_bytes()

    code = _apply(config, _NoLedger(), 0.0731, legs=120, market_open=False, measured={})
    assert code == 2
    assert path.read_bytes() == before


def test_too_few_quoted_legs_is_refused_and_writes_nothing(tmp_path, original):
    from scripts.calibrate_costs import MINIMUM_LEGS, _apply

    path = _write(tmp_path / "convex.yaml", original)
    config = load(path)
    before = path.read_bytes()

    code = _apply(
        config, _NoLedger(), 0.0731, legs=MINIMUM_LEGS - 1, market_open=True, measured={}
    )
    assert code == 2
    assert path.read_bytes() == before


def test_an_open_market_with_a_real_sample_writes_and_leaves_a_receipt(tmp_path, original):
    from scripts.calibrate_costs import _apply

    path = _write(tmp_path / "convex.yaml", original)
    config = load(path)
    ledger = _NoLedger()

    code = _apply(config, ledger, 0.0731, legs=120, market_open=True, measured={"spot": 769.3})
    assert code == 0
    assert KEY not in load(path).hypotheses()
    assert load(path).float_(KEY) == pytest.approx(0.0731)

    # The receipt carries what the threshold was as well as what it became.
    assert len(ledger.records) == 1
    outcome = ledger.records[0].outcome
    assert outcome["after"] == pytest.approx(0.0731)
    assert outcome["before"] != outcome["after"]


class _NoLedger:
    """Collects records instead of writing them."""

    def __init__(self) -> None:
        self.records: list = []

    def append(self, record) -> None:
        self.records.append(record)


def test_a_bound_is_not_something_this_can_clear(original):
    """Only the blocking list is this function's business. A bound is cleared by
    measuring it from a fill, which is a different job entirely."""
    with pytest.raises(ConfigError, match="not listed under provenance.hypothesis"):
        measure(original, key="costs.per_contract_fee")


def test_the_write_replaces_the_file_in_one_step(tmp_path):
    path = tmp_path / "convex.yaml"
    path.write_text("first:\n  value: 1\n")
    write_atomically(path, "second:\n  value: 2\n")
    assert path.read_text() == "second:\n  value: 2\n"
    # No temporary left lying beside it.
    assert [p.name for p in tmp_path.iterdir()] == ["convex.yaml"]


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path
