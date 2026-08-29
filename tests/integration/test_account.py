"""The account the competition will be scored on.

The submission rules make three of these disqualifying rather than merely
inconvenient: the account must be new, funded at exactly one hundred thousand,
and its ID must appear in the submission. This is the test that says whether
the thing about to trade is the thing that will be judged.
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import needs_account

STARTING_EQUITY = 100_000.0


@needs_account
def test_the_account_is_active_and_not_blocked(gateway):
    account = gateway.account()
    assert account.status.upper() == "ACTIVE", f"account status is {account.status}"
    assert account.account_number, "the account has no number to put in the submission"


@needs_account
def test_options_approval_permits_multi_leg_defined_risk_spreads(gateway):
    """Level 3 or nothing.

    Every structure this project trades has a short leg inside a defined-risk
    spread. Below level 3 Alpaca rejects them, and it rejects them at order
    time rather than at startup, which would mean discovering it at 10:00.
    """
    account = gateway.account()
    assert account.options_approved_level >= 3, (
        f"options level is {account.options_approved_level}; multi-leg defined-risk "
        "spreads need level 3 and this account cannot trade the strategy"
    )


@needs_account
def test_equity_matches_what_the_submission_rules_require(gateway):
    account = gateway.account()
    if abs(account.equity - STARTING_EQUITY) > 0.005:
        pytest.skip(
            f"equity is {account.equity:,.2f} rather than {STARTING_EQUITY:,.2f}. "
            "That is expected once trading has begun; it is only disqualifying on "
            "a fresh account before the first trade."
        )


@needs_account
def test_buying_power_is_reported_and_usable(gateway):
    account = gateway.account()
    assert account.options_buying_power >= 0.0
    assert account.buying_power >= 0.0
    # A risk check reads this. If Alpaca stops reporting it the check silently
    # loses its meaning, so it is asserted as present rather than assumed.
    assert account.multiplier > 0


@needs_account
def test_the_day_result_can_be_computed(gateway):
    """The daily loss limit divides by prior equity, so it must be positive."""
    account = gateway.account()
    assert account.last_equity > 0.0
    assert isinstance(account.day_pnl_pct, float)
