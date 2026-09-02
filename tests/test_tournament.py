from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from convex.config import load
from convex.dashboard.app import _tournament_panel, _tournament_rows
from convex.instruments import Right
from convex.ledger import Action, Ledger, Record
from convex.structures.base import Candidate, Family
from convex.tournament import (
    TrendSignal,
    fill_ladder,
    intraday_trend,
    profile_config,
    profiles,
    submission_config,
    trend_candidate_filter,
)
from tests.conftest import leg


def test_profiles_are_isolated_from_main_configuration():
    config = load()
    selected = {profile.name: profile for profile in profiles(config)}
    derived = profile_config(config, selected["skew_bwb"])

    assert derived.list_("structures.enabled") == ["put_bwb", "call_bwb"]
    assert config.list_("structures.enabled") == [
        "put_bwb", "call_bwb", "straddle", "strangle", "debit_vertical"
    ]
    assert derived.path_("paths.ledger").as_posix().endswith("data/tournament/skew_bwb/decisions.jsonl")


def test_fill_ladder_improves_a_credit_without_submitting_it():
    assert fill_ladder(-7.16, [0, 1, 2], 0.01) == [-7.16, -7.15, -7.14]
    assert fill_ladder(1.23, [0, 1, 2], 0.01) == [1.23, 1.24, 1.25]


def test_trend_filter_keeps_only_the_observed_direction():
    bullish = TrendSignal("bullish", 101.0, 100.0, 100.5, 99.0, "test")
    call = Candidate(
        Family.DEBIT_VERTICAL,
        (leg(650, Right.CALL, 2.0, 2.1, 1), leg(655, Right.CALL, 1.0, 1.1, -1)),
        "bull call vertical",
    )
    put = Candidate(
        Family.DEBIT_VERTICAL,
        (leg(650, Right.PUT, 2.0, 2.1, 1), leg(645, Right.PUT, 1.0, 1.1, -1)),
        "bear put vertical",
    )

    allowed = trend_candidate_filter(bullish)
    assert allowed(call)
    assert not allowed(put)


class _BarsGateway:
    def __init__(self, frame):
        self.frame = frame

    def minute_bars(self, symbol, start, end):
        assert symbol == "SPY"
        return self.frame


def test_intraday_trend_requires_range_break_and_vwap_agreement():
    config = load()
    index = pd.date_range("2026-09-02T13:30:00Z", periods=16, freq="min")
    frame = pd.DataFrame({
        "close": [100.0] * 15 + [101.0],
        "high": [100.2] * 15 + [101.1],
        "low": [99.8] * 16,
        "volume": [100] * 16,
    }, index=index)
    now = datetime(2026, 9, 2, 13, 46, tzinfo=UTC)

    signal = intraday_trend(_BarsGateway(frame), config, now)

    assert signal.direction == "bullish"
    assert signal.tradeable


def test_dashboard_keeps_tournament_dry_runs_out_of_account_pnl(tmp_path):
    config = load().with_overrides({
        "tournament.profiles.skew_bwb.ledger": str(tmp_path / "skew.jsonl"),
    })
    ledger = Ledger(config.path_("tournament.profiles.skew_bwb.ledger"))
    ledger.append(Record(
        action=Action.DRY_RUN,
        cycle_id="tournament-test",
        structure="call_bwb",
        rationale="Would open only; no broker order was sent.",
    ))

    rows = _tournament_rows(config)
    skew = next(row for row in rows if row["name"] == "skew_bwb")

    assert skew["dry_runs"] == 1
    assert skew["verified_fills"] == 0
    assert skew["realised_pnl"] == 0.0
    page = _tournament_panel(rows)
    assert "do not change account P&amp;L" in page
    assert "Skew BWB" in page


def test_submission_config_is_one_structure_and_uses_the_canonical_ledger():
    config = load()
    selected = {profile.name: profile for profile in profiles(config)}

    submitted = submission_config(config, selected["skew_bwb"])

    assert submitted.int_("risk.max_concurrent_structures") == 1
    assert submitted.path_("paths.ledger") == config.path_("paths.ledger")
    assert submitted.list_("structures.enabled") == ["put_bwb", "call_bwb"]


def test_only_bwb_profiles_are_submission_eligible():
    import pytest

    from convex.errors import DataError

    config = load()
    selected = {profile.name: profile for profile in profiles(config)}
    assert submission_config(config, selected["execution_bwb"]).int_("risk.max_concurrent_structures") == 1
    with pytest.raises(DataError, match="observation-only"):
        submission_config(config, selected["trend_vertical"])
