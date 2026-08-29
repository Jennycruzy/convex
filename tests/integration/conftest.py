"""Fixtures that talk to the real paper account.

Nothing in this directory is mocked. Every test here opens a real connection to
Alpaca's MCP server, reads real market data, and asserts against what actually
came back. They skip cleanly when credentials are absent so the unit suite
stays runnable on a laptop, and they are the tests that decide whether the
agent is allowed to trade on a given morning.

They never place an order. The execution path is exercised through the same
dry-run wrapper the live script uses, which withholds only the write.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from convex.config import load

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _credentials_present() -> bool:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    return bool(
        os.environ.get("ALPACA_API_KEY", "").strip()
        and os.environ.get("ALPACA_SECRET_KEY", "").strip()
    )


needs_account = pytest.mark.skipif(
    not _credentials_present(),
    reason="no Alpaca paper credentials in .env; these tests talk to the live account",
)


@pytest.fixture(scope="session")
def config():
    return load()


@pytest.fixture(scope="session")
def gateway(config):
    """One MCP connection shared by the whole session.

    Shared deliberately: each connection spawns a server subprocess, and a
    fixture per test would spend more time starting servers than testing.
    """
    from convex.data.alpaca import AlpacaGateway

    connection = AlpacaGateway(config)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def spot(gateway, config):
    price, _ = gateway.spot(config.str_("underlying.symbol"))
    return price


@pytest.fixture(scope="session")
def today_expiry(gateway, config):
    """The nearest listed expiry, and whether it is actually today."""
    now, _ = gateway.clock()
    expiries = gateway.expirations(config.str_("underlying.symbol"), now.date())
    return expiries[0], expiries[0] == now.date()


@pytest.fixture(scope="session")
def chain(gateway, config, spot, today_expiry):
    """The real 0DTE chain in the candidate band."""
    expiry, _ = today_expiry
    low = spot * config.float_("candidates.moneyness_low")
    high = spot * config.float_("candidates.moneyness_high")
    wing = spot * config.float_("candidates.max_wing_width_pct")
    return gateway.chain(config.str_("underlying.symbol"), expiry, low - wing, high + wing)
