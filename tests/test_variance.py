"""The implied variance history the regime rule is handed.

Every test here is about a refusal. The rule is the only thing choosing a
direction while no model ships, so a history that is stale, thin, for another
symbol, or contaminated by the session being decided is a wrong decision with a
receipt that reads exactly like a right one.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from convex.errors import DataError
from convex.variance import load_history


def write(path, series, symbol="SPY"):
    path.write_text(
        json.dumps(
            {
                "built_at": "2026-09-01T23:08:14-04:00",
                "symbol": symbol,
                "entry_time": "10:00:00",
                "close_time": "16:00:00",
                "sessions": len(series),
                "series": series,
            }
        )
    )
    return path


def sessions(count, start=date(2026, 1, 1), reading=0.05):
    return [
        {
            "session": date.fromordinal(start.toordinal() + index).isoformat(),
            "iv_total": reading + index * 0.0001,
            "coverage": 0.6,
        }
        for index in range(count)
    ]


def test_it_reads_the_readings_in_session_order(tmp_path):
    rows = sessions(40)
    history = load_history(write(tmp_path / "iv.json", rows), "SPY", date(2026, 2, 10), 5, 30)

    assert len(history) == 40
    assert history.first_session == date(2026, 1, 1)
    assert history.last_session == date(2026, 2, 9)
    assert history.as_list() == sorted(history.as_list())


def test_the_session_being_decided_never_enters_its_own_history(tmp_path):
    """A rebuild that has caught up to today would otherwise hand over the answer."""
    rows = sessions(40, start=date(2026, 1, 1))
    today = date.fromisoformat(rows[-1]["session"])

    history = load_history(write(tmp_path / "iv.json", rows), "SPY", today, 5, 30)

    assert len(history) == 39
    assert history.last_session < today


def test_a_stale_history_raises_rather_than_being_compared_against(tmp_path):
    rows = sessions(40, start=date(2026, 1, 1))

    with pytest.raises(DataError, match="staleness budget"):
        load_history(write(tmp_path / "iv.json", rows), "SPY", date(2026, 3, 1), 5, 30)


def test_too_few_readings_to_take_a_quantile_from_raises(tmp_path):
    rows = sessions(12, start=date(2026, 1, 1))

    with pytest.raises(DataError, match="at least 30"):
        load_history(write(tmp_path / "iv.json", rows), "SPY", date(2026, 1, 13), 5, 30)


def test_a_history_for_another_symbol_raises(tmp_path):
    rows = sessions(40)

    with pytest.raises(DataError, match="QQQ"):
        load_history(write(tmp_path / "iv.json", rows, symbol="QQQ"), "SPY", date(2026, 2, 10), 5, 30)


def test_a_missing_history_names_the_script_that_builds_it(tmp_path):
    with pytest.raises(DataError, match="variance_history"):
        load_history(tmp_path / "absent.json", "SPY", date(2026, 2, 10), 5, 30)


def test_a_non_positive_reading_is_not_a_variance(tmp_path):
    rows = sessions(40)
    rows[7]["iv_total"] = 0.0

    with pytest.raises(DataError, match="not a variance"):
        load_history(write(tmp_path / "iv.json", rows), "SPY", date(2026, 2, 10), 5, 30)


def test_the_live_history_on_disk_answers_for_today():
    """The file the 10:00 cycle will actually read, checked as it is."""
    from convex.config import load

    config = load()
    path = config.path_("paths.variance_history")
    if not path.exists():
        pytest.skip("no rebuilt history on this machine")

    history = load_history(
        path,
        config.str_("underlying.symbol"),
        date.today(),
        config.int_("classifier.variance_history_max_age_days"),
        config.int_("classifier.variance_history_min_readings"),
    )
    assert len(history) >= config.int_("classifier.variance_history_min_readings")
    assert history.last_session < date.today()
