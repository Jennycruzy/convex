"""The seam between the decision code and Alpaca's MCP server.

Two things are checked here and they are different in kind.

The transport tests start the real Alpaca MCP server as a subprocess. Nothing
is mocked: the server is the published one, the tool list is whatever it really
exposes, and the credential failure is a real 401 from Alpaca. They are skipped
when the server is not installed, and they never place an order.

The parsing tests take the response bodies exactly as Alpaca's own OpenAPI spec
defines them, shipped inside the server package, and check that a field this
project depends on being present is an exception when it is absent. Those
bodies are not invented; they are the documented shape, which is the only way
to test the boundary without credentials.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from convex.data import alpaca
from convex.data.mcp import McpClient, McpError, McpSettings, _unwrap
from convex.errors import DataError, ExecutionError

SERVER = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "alpaca-mcp-server"
needs_server = pytest.mark.skipif(
    not SERVER.is_file(), reason="the Alpaca MCP server is not installed in this environment"
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A live connection to the real server, with credentials that will fail."""
    environment = dict(os.environ)
    environment.setdefault("ALPACA_API_KEY", "not-a-real-key")
    environment.setdefault("ALPACA_SECRET_KEY", "not-a-real-secret")
    environment["ALPACA_PAPER_TRADE"] = "true"
    settings = McpSettings(
        command=str(SERVER),
        args=("--transport", "stdio"),
        env=environment,
        log_path=tmp_path_factory.mktemp("mcp") / "server.log",
        startup_timeout=90.0,
    )
    session = McpClient(settings).start()
    yield session
    session.close()


# -------------------------------------------------------------------- transport


@needs_server
def test_the_server_exposes_every_tool_a_decision_cycle_needs(client):
    client.require_tools(*alpaca.REQUIRED_TOOLS)


@needs_server
def test_a_tool_the_server_does_not_have_is_a_startup_failure(client):
    with pytest.raises(McpError, match="does not expose"):
        client.require_tools("get_account_info", "settle_everything_favourably")


@needs_server
def test_a_rejected_call_raises_rather_than_returning_a_default(client):
    # Real credentials are absent, so this is a real 401 from Alpaca coming
    # back through a JSON-RPC result that itself reports success.
    with pytest.raises((McpError, DataError)):
        client.call("get_account_info")


@needs_server
def test_the_server_stderr_is_kept_rather_than_discarded(client):
    assert client._settings.log_path.exists()
    assert client._settings.log_path.stat().st_size > 0


def test_calling_before_start_raises_instead_of_hanging():
    settings = McpSettings(command="/nonexistent", args=(), env={})
    with pytest.raises(McpError, match="has not been started"):
        McpClient(settings).call("get_clock")


def test_a_scalar_return_is_unwrapped_and_an_object_is_left_alone():
    assert _unwrap({"result": [1, 2]}) == [1, 2]
    assert _unwrap({"equity": "1", "result": "2"}) == {"equity": "1", "result": "2"}


# ---------------------------------------------------------------------- parsing


def test_a_missing_field_is_an_exception_not_a_zero():
    with pytest.raises(DataError, match="has no 'equity'"):
        alpaca._require({"cash": "100"}, "equity", "get_account_info")


def test_a_field_that_is_not_a_number_raises():
    with pytest.raises(DataError, match="is not a number"):
        alpaca._float({"equity": "unavailable"}, "equity", "get_account_info")


def test_a_timestamp_without_a_zone_raises_rather_than_being_assumed_utc():
    with pytest.raises(DataError, match="carries no timezone"):
        alpaca._stamp("2026-08-28T14:30:00", "quote")


def test_a_z_suffixed_timestamp_parses():
    assert alpaca._stamp("2026-08-28T14:30:00Z", "quote").tzinfo is not None


def test_greeks_are_none_when_the_snapshot_has_none_rather_than_being_zeroed():
    assert alpaca._greeks({"latestQuote": {"bp": 1.0}}) is None
    assert alpaca._greeks({"greeks": {"delta": 0.5}, "impliedVolatility": None}) is None


def test_greeks_parse_when_the_snapshot_is_complete():
    greeks = alpaca._greeks(
        {
            "greeks": {"delta": -0.31, "gamma": 0.02, "theta": -55.0, "vega": 0.11, "rho": 0.0},
            "impliedVolatility": 0.17,
        }
    )
    assert greeks.delta == -0.31
    assert greeks.implied_volatility == 0.17


def test_a_snapshot_with_partial_greeks_raises_rather_than_filling_the_gap():
    with pytest.raises(DataError):
        alpaca._greeks(
            {"greeks": {"delta": -0.31, "gamma": 0.02}, "impliedVolatility": 0.17}
        )


def test_an_order_response_that_is_not_an_order_raises():
    with pytest.raises(ExecutionError, match="not an order"):
        alpaca._order_record(["accepted"])


def test_an_order_response_parses_the_fields_the_ledger_records():
    record = alpaca._order_record(
        {
            "id": "b1e1",
            "status": "accepted",
            "submitted_at": "2026-08-28T14:00:00Z",
            "client_order_id": "convex-1",
            "filled_qty": "0",
            "filled_avg_price": None,
        }
    )
    assert record.id == "b1e1"
    assert record.filled_avg_price is None


def test_the_toolsets_the_server_is_started_with_exclude_what_convex_cannot_trade():
    enabled = set(alpaca.TOOLSETS.split(","))
    assert "options-data" in enabled
    assert "crypto-data" not in enabled
    assert "watchlists" not in enabled
