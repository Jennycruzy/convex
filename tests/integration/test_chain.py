"""The 0DTE chain, as it really comes back.

This is the Day-1 acceptance test. Every parameter carried over from the SPX
research is a hypothesis until it has been measured on SPY, and this is where
the measuring happens: strike increments, multipliers, how many contracts carry
Greeks, and above all what the per-leg spread actually is, because the whole
cost argument rests on that number.
"""

from __future__ import annotations

import statistics

import pytest

from convex.instruments import Right
from tests.integration.conftest import needs_account


@needs_account
def test_a_same_day_expiry_exists_or_the_reason_is_stated(today_expiry, gateway):
    expiry, is_today = today_expiry
    now, _ = gateway.clock()
    if not is_today:
        pytest.skip(
            f"the nearest listed expiry is {expiry}, not {now.date()}. This project "
            "trades same-day expiries only, so there is no cycle today."
        )


@needs_account
def test_the_chain_carries_greeks_and_implied_volatility(chain, config):
    """Without these the feature engine cannot run at all."""
    with_greeks = [row for row in chain if row.greeks is not None]
    share = len(with_greeks) / len(chain)
    print(f"  {len(with_greeks)} of {len(chain)} contracts carry Greeks ({share:.0%})")
    assert with_greeks, (
        f"the {config.str_('data.options_feed')} feed returned no Greeks; the "
        "classifier's strongest feature cannot be computed"
    )
    for row in with_greeks[:20]:
        greeks = row.require_greeks()
        assert greeks.implied_volatility > 0.0
        assert -1.0 <= greeks.delta <= 1.0
        assert greeks.gamma >= 0.0


@needs_account
def test_open_interest_comes_back_for_the_exposure_features(chain):
    with_oi = [row for row in chain if row.open_interest is not None]
    print(f"  {len(with_oi)} of {len(chain)} contracts carry open interest")
    assert with_oi, "no open interest came back, so the exposure proxies cannot be built"


@needs_account
def test_the_strike_increment_is_a_dollar_as_the_translation_assumes(chain):
    """SPX moves in fives and SPY in ones. The candidate builder assumes ones."""
    strikes = sorted({row.contract.strike for row in chain if row.contract.right is Right.PUT})
    gaps = {round(b - a, 4) for a, b in zip(strikes, strikes[1:])}
    print(f"  strike gaps observed: {sorted(gaps)}")
    assert gaps, "fewer than two put strikes came back"
    assert min(gaps) <= 1.0, f"the narrowest strike gap is {min(gaps)}, not the expected 1.00"


@needs_account
def test_the_contract_multiplier_is_measured_rather_than_assumed(chain, config):
    multipliers = {row.contract.multiplier for row in chain}
    print(f"  multipliers observed: {sorted(multipliers)}")
    assert len(multipliers) == 1, f"mixed multipliers in one chain: {sorted(multipliers)}"
    configured = config.int_("underlying.contract_multiplier")
    assert multipliers == {configured}, (
        f"the chain reports a multiplier of {multipliers.pop()} but the configuration "
        f"says {configured}; every max loss in the system would be wrong"
    )


@needs_account
def test_the_per_leg_spread_is_measured_and_recorded(chain, config):
    """The number the entire cost argument rests on.

    Printed rather than merely asserted, because the measured value is what
    goes into the configuration and into the write-up. A threshold that has
    never been compared against a real book is a guess with a colon after it.
    """
    relative = sorted(row.quote.relative_spread for row in chain)
    half = sorted(row.quote.half_spread for row in chain)
    median_relative = statistics.median(relative)

    print(f"  relative spread : median {median_relative:.1%}, "
          f"p90 {relative[int(len(relative) * 0.9)]:.1%}, widest {relative[-1]:.1%}")
    print(f"  half spread     : median {statistics.median(half):.3f}, widest {half[-1]:.3f}")
    print(f"  four-leg structure pays roughly "
          f"{4 * statistics.median(half) * 100:.2f} dollars per lot to enter at mid")

    threshold = config.float_("liquidity.max_relative_spread")
    tradable = [value for value in relative if value <= threshold]
    print(f"  {len(tradable)} of {len(relative)} legs pass the {threshold:.0%} liquidity limit")
    assert tradable, (
        f"every leg in the band is wider than the {threshold:.0%} liquidity limit; "
        "either the limit is wrong or the book is unusable right now"
    )


@needs_account
def test_quotes_are_fresh_enough_for_the_staleness_budget(chain, config, gateway):
    now, _ = gateway.clock()
    budget = config.float_("liquidity.max_quote_age_seconds")
    ages = sorted(row.quote.age_seconds(now) for row in chain)
    print(f"  quote age: median {statistics.median(ages):.1f}s, oldest {ages[-1]:.1f}s "
          f"against a {budget:.0f}s budget")
    assert ages[0] < budget * 10, "every quote in the chain is far outside the budget"


@needs_account
def test_no_quote_in_the_chain_is_crossed_or_unpriceable(chain):
    """Quote construction raises on these, so arriving here means they parsed."""
    for row in chain:
        assert row.quote.ask >= row.quote.bid
        assert row.quote.mid > 0.0
