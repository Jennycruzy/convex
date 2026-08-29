"""What Alpaca's own calendar says about the competition window.

The build specification's day-by-day plan labels 29 August as a Friday and
1 September as Labor Day. Neither is true of 2026: 29 August is a Saturday and
Labor Day falls on 7 September. A local holiday table is exactly how that kind
of error survives, so the market-calendar check reads Alpaca's endpoint and
this test asserts against it rather than against the plan.
"""

from __future__ import annotations

from datetime import date, timedelta

from tests.integration.conftest import needs_account

COMPETITION_START = date(2026, 8, 28)
COMPETITION_END = date(2026, 9, 4)


@needs_account
def test_the_clock_comes_from_alpaca_not_from_this_machine(gateway):
    now, is_open = gateway.clock()
    assert now.tzinfo is not None, "the exchange clock must carry a timezone"
    assert isinstance(is_open, bool)


@needs_account
def test_the_competition_window_holds_the_sessions_the_plan_depends_on(gateway):
    """Print and check every session between the start and the deadline."""
    sessions = gateway.sessions(COMPETITION_START, COMPETITION_END)
    open_days = {session.session_date for session in sessions}

    weekdays = {
        COMPETITION_START + timedelta(days=offset)
        for offset in range((COMPETITION_END - COMPETITION_START).days + 1)
        if (COMPETITION_START + timedelta(days=offset)).weekday() < 5
    }
    closed_weekdays = sorted(weekdays - open_days)

    for session in sessions:
        print(f"  {session.session_date} {session.session_date:%a}  "
              f"{session.open_at:%H:%M}-{session.close_at:%H:%M}")
    for day in closed_weekdays:
        print(f"  {day} {day:%a} is a weekday with no session")

    assert sessions, "Alpaca reports no sessions at all in the competition window"
    # Weekends must not appear. This is the assertion that would have caught the
    # plan's first-live-trade-on-Saturday instruction.
    assert all(session.session_date.weekday() < 5 for session in sessions)


@needs_account
def test_a_saturday_is_not_a_trading_day(gateway):
    assert not gateway.is_trading_day(date(2026, 8, 29))


@needs_account
def test_sessions_close_at_the_regular_bell_not_the_extended_one(gateway):
    """The research holds to the 16:00 close, so the close must be 16:00."""
    sessions = gateway.sessions(COMPETITION_START, COMPETITION_END)
    for session in sessions:
        assert session.close_at.hour in (13, 16), (
            f"{session.session_date} closes at {session.close_at:%H:%M}; a 13:00 close "
            "is a half day and the entry logic must know about it"
        )
        assert session.open_at < session.close_at
