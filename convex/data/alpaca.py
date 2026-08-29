"""The market, reached through Alpaca's MCP server.

Every fact this project acts on arrives here: the account, the clock, the
calendar, the 0DTE chain with its Greeks and open interest, the quotes each
candidate is priced against, the orders, and the positions the guard watches.
Nothing above this module talks to a network, and nothing below it makes a
decision.

The transport is the Model Context Protocol. The server is the one Alpaca
publishes, run as a child process over a pipe, and its tools are the same
endpoints the REST API exposes. That has one consequence worth stating: MCP
tools answer with JSON, not with typed SDK objects, so this module is where
strings become floats, where a missing field becomes an exception, and where
Law 3 is enforced at the boundary. A quote that cannot be parsed does not
become a zero here; it raises, and the cycle that wanted it stops.

The gateway hands back the same objects it always did, ChainEntry and Quote
and Greeks and AccountSnapshot, so the decision code neither knows nor cares which
transport it is on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from convex.config import Config
from convex.data.mcp import McpClient, McpSettings
from convex.errors import CredentialsError, DataError, ExecutionError
from convex.instruments import ChainEntry, Greeks, Leg, OptionContract, Quote, Right

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The tools a decision cycle cannot run without. Checked once at startup so a
# server that has renamed or dropped one fails before 10:00, not during it.
REQUIRED_TOOLS = (
    "get_account_info",
    "get_clock",
    "get_calendar",
    "get_option_contracts",
    "get_option_snapshot",
    "get_stock_latest_quote",
    "get_stock_bars",
    "get_all_positions",
    "get_order_by_id",
    "place_option_order",
)

# Only these toolsets are enabled on the server. CONVEX trades one ETF's
# options; a server exposing crypto orders and watchlists to the same session
# is surface this project has no use for.
TOOLSETS = "account,trading,assets,stock-data,options-data"

_MAX_BARS_PER_PAGE = 10_000

# A 2% band on SPY is around sixty contracts once both rights are counted, and
# the request goes in a query string, so the symbol list is sent in batches.
_SYMBOLS_PER_REQUEST = 40


def _batched(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


@dataclass(frozen=True)
class AccountSnapshot:
    """The account as Alpaca reports it. Nothing here is assumed."""

    account_number: str
    status: str
    equity: float
    last_equity: float
    cash: float
    buying_power: float
    options_buying_power: float
    options_approved_level: int
    multiplier: float
    pattern_day_trader: bool

    @property
    def day_pnl(self) -> float:
        return self.equity - self.last_equity

    @property
    def day_pnl_pct(self) -> float:
        if self.last_equity <= 0.0:
            raise DataError(f"account reports non-positive prior equity {self.last_equity}")
        return self.day_pnl / self.last_equity


@dataclass(frozen=True)
class MarketSession:
    """One trading day from Alpaca's calendar, in exchange local time."""

    session_date: date
    open_at: datetime
    close_at: datetime


@dataclass(frozen=True)
class PositionRecord:
    """One open position. The guard reads these and closes what it must."""

    symbol: str
    qty: str
    asset_class: str
    avg_entry_price: str
    unrealized_pl: str
    market_value: str


@dataclass(frozen=True)
class OrderRecord:
    """An order as Alpaca acknowledged it."""

    id: str
    status: str
    submitted_at: str
    client_order_id: str
    filled_qty: str
    filled_avg_price: str | None


# --------------------------------------------------------------------- parsing


def _require(payload: dict, key: str, context: str) -> Any:
    if key not in payload or payload[key] is None:
        raise DataError(f"{context}: the response has no {key!r} (keys: {sorted(payload)})")
    return payload[key]


def _float(payload: dict, key: str, context: str) -> float:
    value = _require(payload, key, context)
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise DataError(f"{context}: {key}={value!r} is not a number") from error


def _stamp(value: str, context: str) -> datetime:
    """Parse an RFC 3339 timestamp, keeping its zone rather than assuming one."""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise DataError(f"{context}: {value!r} is not a timestamp") from error
    if parsed.tzinfo is None:
        raise DataError(f"{context}: {value!r} carries no timezone")
    return parsed


class AlpacaGateway:
    """A single MCP connection to the paper account and its market data."""

    def __init__(
        self,
        config: Config,
        env_path: Path | None = None,
        client: McpClient | None = None,
    ) -> None:
        self.config = config
        self._options_feed = config.str_("data.options_feed")
        self._stock_feed = config.str_("data.stock_feed")
        self._zone = ZoneInfo(config.str_("session.timezone"))
        self._client = client if client is not None else self._spawn(env_path)
        self._client.require_tools(*REQUIRED_TOOLS)

    def _spawn(self, env_path: Path | None) -> McpClient:
        load_dotenv(env_path or REPO_ROOT / ".env")
        key = os.environ.get("ALPACA_API_KEY", "").strip()
        secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
        if not key or not secret:
            raise CredentialsError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set. Copy .env.example to "
                ".env and fill in the paper account's keys; .env is gitignored."
            )
        paper = os.environ.get("ALPACA_PAPER", "true").strip().lower()
        if paper not in {"true", "false"}:
            raise CredentialsError(f"ALPACA_PAPER must be true or false, found {paper!r}")
        if paper == "false":
            raise CredentialsError(
                "CONVEX is a paper-account project and refuses to start against live money"
            )

        command = os.environ.get("ALPACA_MCP_COMMAND", "").strip()
        if not command:
            candidate = Path(REPO_ROOT / ".venv" / "bin" / "alpaca-mcp-server")
            if not candidate.is_file():
                raise CredentialsError(
                    "the Alpaca MCP server was not found. Install it with "
                    "`uv pip install alpaca-mcp-server` or set ALPACA_MCP_COMMAND "
                    "to its path."
                )
            command = str(candidate)

        settings = McpSettings(
            command=command,
            args=("--transport", "stdio"),
            env={
                **os.environ,
                "ALPACA_API_KEY": key,
                "ALPACA_SECRET_KEY": secret,
                "ALPACA_PAPER_TRADE": "true",
                "ALPACA_TOOLSETS": TOOLSETS,
            },
            log_path=self.config.path_("paths.mcp_log"),
        )
        return McpClient(settings).start()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AlpacaGateway":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------ account

    def account(self) -> AccountSnapshot:
        raw = self._client.call("get_account_info")
        context = "get_account_info"
        if not isinstance(raw, dict):
            raise DataError(f"{context} returned {type(raw).__name__}, not an account")
        return AccountSnapshot(
            account_number=str(_require(raw, "account_number", context)),
            status=str(_require(raw, "status", context)),
            equity=_float(raw, "equity", context),
            last_equity=_float(raw, "last_equity", context),
            cash=_float(raw, "cash", context),
            buying_power=_float(raw, "buying_power", context),
            options_buying_power=_float(raw, "options_buying_power", context),
            options_approved_level=int(_require(raw, "options_approved_level", context)),
            multiplier=_float(raw, "multiplier", context),
            pattern_day_trader=bool(raw.get("pattern_day_trader", False)),
        )

    def clock(self) -> tuple[datetime, bool]:
        """Exchange time and whether the market is open, from Alpaca not locally."""
        raw = self._client.call("get_clock")
        stamp = _stamp(_require(raw, "timestamp", "get_clock"), "get_clock timestamp")
        return stamp.astimezone(self._zone), bool(_require(raw, "is_open", "get_clock"))

    def sessions(self, start: date, end: date) -> list[MarketSession]:
        """Trading sessions in the window. A closed day simply does not appear."""
        raw = self._client.call(
            "get_calendar", {"start": start.isoformat(), "end": end.isoformat()}
        )
        if not isinstance(raw, list):
            raise DataError(f"get_calendar returned {type(raw).__name__}, not a list of days")
        sessions: list[MarketSession] = []
        for day in raw:
            session_date = date.fromisoformat(str(_require(day, "date", "get_calendar")))
            sessions.append(
                MarketSession(
                    session_date=session_date,
                    open_at=self._exchange_time(session_date, str(_require(day, "open", "get_calendar"))),
                    close_at=self._exchange_time(session_date, str(_require(day, "close", "get_calendar"))),
                )
            )
        return sorted(sessions, key=lambda session: session.session_date)

    def _exchange_time(self, day: date, clock: str) -> datetime:
        """Alpaca's calendar gives local wall-clock times; attach the zone."""
        hour, _, minute = clock.partition(":")
        try:
            return datetime.combine(day, time(int(hour), int(minute)), tzinfo=self._zone)
        except ValueError as error:
            raise DataError(f"the calendar gave {clock!r} as a session time") from error

    def is_trading_day(self, day: date) -> bool:
        return any(session.session_date == day for session in self.sessions(day, day))

    # -------------------------------------------------------------- stock data

    def spot(self, symbol: str) -> tuple[float, datetime]:
        """The underlying's mid and the time it was quoted."""
        raw = self._client.call(
            "get_stock_latest_quote", {"symbols": symbol, "feed": self._stock_feed}
        )
        quotes = _require(raw, "quotes", "get_stock_latest_quote")
        if symbol not in quotes:
            raise DataError(f"get_stock_latest_quote returned no quote for {symbol}")
        entry = quotes[symbol]
        context = f"{symbol} spot quote"
        bid = _float(entry, "bp", context)
        ask = _float(entry, "ap", context)
        if bid <= 0.0 or ask <= 0.0:
            raise DataError(f"{context}: one-sided quote bid={bid} ask={ask}")
        if ask < bid:
            raise DataError(f"{context}: crossed quote bid={bid} ask={ask}")
        return (bid + ask) / 2.0, _stamp(_require(entry, "t", context), context)

    def minute_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Minute bars for the underlying, used to build the return history.

        The tool caps a page at ten thousand points and a two-year minute
        history is thirty times that, so the history has to be collected in
        pieces. Unlike the contract and option-bar tools, this one takes no
        page token: the server (3.4.7) rejects the argument outright rather
        than ignoring it. So the walk is done on time instead, carrying the
        start forward to the minute after the last bar of the page just read
        and asking again. Every page is kept; none is sampled or thinned.

        A page that comes back full but does not move the clock forward would
        spin here forever, so that raises rather than looping.
        """
        rows: list[dict] = []
        cursor = start
        while cursor < end:
            arguments: dict[str, Any] = {
                "symbols": symbol,
                "timeframe": "1Min",
                "start": cursor.isoformat(),
                "end": end.isoformat(),
                "limit": _MAX_BARS_PER_PAGE,
                "feed": self._stock_feed,
                "sort": "asc",
            }
            raw = self._client.call("get_stock_bars", arguments)
            bars = _require(raw, "bars", "get_stock_bars")
            page = bars.get(symbol) or []
            if not page:
                break
            rows.extend(page)
            if len(page) < _MAX_BARS_PER_PAGE:
                break
            last = _stamp(_require(page[-1], "t", "minute bar"), "minute bar")
            if last <= cursor:
                raise DataError(
                    f"get_stock_bars returned a full page for {symbol} ending at "
                    f"{last.isoformat()}, which does not advance past {cursor.isoformat()}"
                )
            cursor = last + timedelta(minutes=1)

        if not rows:
            raise DataError(
                f"Alpaca returned no minute bars for {symbol} between {start} and {end}"
            )
        frame = pd.DataFrame(rows)
        missing = {"t", "c"} - set(frame.columns)
        if missing:
            raise DataError(f"minute bars are missing {sorted(missing)}: {list(frame.columns)}")
        frame["timestamp"] = pd.to_datetime(frame["t"], utc=True, format="ISO8601")
        frame = frame.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        return frame.set_index("timestamp").sort_index()

    # ------------------------------------------------------------------ chains

    def expirations(self, symbol: str, on_or_before: date) -> list[date]:
        """Expiries Alpaca actually lists for this underlying, soonest first."""
        contracts = self._contract_pages(
            {
                "underlying_symbols": symbol,
                "expiration_date_gte": on_or_before.isoformat(),
                "expiration_date_lte": (on_or_before + timedelta(days=14)).isoformat(),
                "status": "active",
            }
        )
        if not contracts:
            raise DataError(f"Alpaca lists no {symbol} contracts expiring near {on_or_before}")
        return sorted(
            {date.fromisoformat(str(row["expiration_date"])) for row in contracts}
        )

    def chain(
        self,
        symbol: str,
        expiry: date,
        strike_low: float,
        strike_high: float,
    ) -> list[ChainEntry]:
        """Every tradable contract in the strike band, priced and with Greeks."""
        if strike_high <= strike_low:
            raise DataError(f"strike band is inverted: {strike_low} to {strike_high}")

        listed = self._list_contracts(symbol, expiry, strike_low, strike_high)
        snapshots = self._snapshots(list(listed))

        rows: list[ChainEntry] = []
        for occ_symbol, (contract, open_interest) in listed.items():
            snapshot = snapshots.get(occ_symbol)
            if snapshot is None or not snapshot.get("latestQuote"):
                continue
            raw_quote = snapshot["latestQuote"]
            context = f"{occ_symbol} quote"
            bid = _float(raw_quote, "bp", context)
            ask = _float(raw_quote, "ap", context)
            if bid <= 0.0 and ask <= 0.0:
                continue
            rows.append(
                ChainEntry(
                    contract=contract,
                    quote=Quote(
                        symbol=occ_symbol,
                        bid=bid,
                        ask=ask,
                        bid_size=int(_require(raw_quote, "bs", context)),
                        ask_size=int(_require(raw_quote, "as", context)),
                        timestamp=_stamp(_require(raw_quote, "t", context), context),
                    ),
                    greeks=_greeks(snapshot),
                    open_interest=open_interest,
                    volume=_volume(snapshot),
                )
            )
        if not rows:
            raise DataError(
                f"no priceable {symbol} contracts for {expiry} between "
                f"{strike_low} and {strike_high} on the {self._options_feed} feed"
            )
        return sorted(rows, key=lambda row: (row.contract.right, row.contract.strike))

    def _contract_pages(self, arguments: dict[str, Any]) -> list[dict]:
        """Every page of a contract listing, followed to the end."""
        collected: list[dict] = []
        page_token: str | None = None
        while True:
            payload = dict(arguments, limit=10_000)
            if page_token:
                payload["page_token"] = page_token
            raw = self._client.call("get_option_contracts", payload)
            collected.extend(raw.get("option_contracts") or [])
            page_token = raw.get("next_page_token")
            if not page_token:
                break
        return collected

    def _list_contracts(
        self, symbol: str, expiry: date, strike_low: float, strike_high: float
    ) -> dict[str, tuple[OptionContract, int | None]]:
        """Tradable contracts in the band, each with the open interest Alpaca
        publishes for it. Open interest is a trading-API field, not a market-data
        one, which is why the chain is assembled from both."""
        contracts: dict[str, tuple[OptionContract, int | None]] = {}
        for raw in self._contract_pages(
            {
                "underlying_symbols": symbol,
                "expiration_date": expiry.isoformat(),
                "strike_price_gte": strike_low,
                "strike_price_lte": strike_high,
                "status": "active",
            }
        ):
            if not raw.get("tradable"):
                continue
            occ_symbol = str(_require(raw, "symbol", "get_option_contracts"))
            context = f"{occ_symbol} contract"
            open_interest = raw.get("open_interest")
            contracts[occ_symbol] = (
                OptionContract(
                    symbol=occ_symbol,
                    underlying=str(_require(raw, "underlying_symbol", context)),
                    right=Right.CALL if str(raw.get("type")).lower() == "call" else Right.PUT,
                    strike=_float(raw, "strike_price", context),
                    expiry=date.fromisoformat(str(_require(raw, "expiration_date", context))),
                    multiplier=int(_float(raw, "size", context)),
                ),
                int(open_interest) if open_interest is not None else None,
            )
        if not contracts:
            raise DataError(
                f"Alpaca lists no tradable {symbol} contracts expiring {expiry} "
                f"between strikes {strike_low} and {strike_high}"
            )
        return contracts

    def _snapshots(self, symbols: list[str]) -> dict[str, dict]:
        """Quotes, Greeks and implied volatility, batched by the tool's limit."""
        snapshots: dict[str, dict] = {}
        batch_size = 100
        for start in range(0, len(symbols), batch_size):
            batch = symbols[start : start + batch_size]
            raw = self._client.call(
                "get_option_snapshot",
                {"symbols": ",".join(batch), "feed": self._options_feed},
            )
            snapshots.update(_require(raw, "snapshots", "get_option_snapshot"))
        return snapshots

    def option_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        """Live two-sided quotes for contracts already held.

        The chain builder cannot serve this: it filters to a strike band around
        spot, and a leg the guard needs to close may have moved outside that
        band, which is precisely the case where closing it matters most.
        """
        if not symbols:
            return {}
        quotes: dict[str, Quote] = {}
        for occ_symbol, snapshot in self._snapshots(list(symbols)).items():
            raw_quote = snapshot.get("latestQuote")
            if not raw_quote:
                continue
            context = f"{occ_symbol} quote"
            quotes[occ_symbol] = Quote(
                symbol=occ_symbol,
                bid=_float(raw_quote, "bp", context),
                ask=_float(raw_quote, "ap", context),
                bid_size=int(_require(raw_quote, "bs", context)),
                ask_size=int(_require(raw_quote, "as", context)),
                timestamp=_stamp(_require(raw_quote, "t", context), context),
            )
        missing = set(symbols) - set(quotes)
        if missing:
            raise DataError(f"no quote came back for {', '.join(sorted(missing))}")
        return quotes

    def option_bars(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> dict[str, list[dict]]:
        """Traded bars per contract, for rebuilding sessions that were not recorded.

        This is prints, not quotes. Alpaca keeps no historical option book, so a
        past session can only be rebuilt from what actually traded, and a print
        is not a price anyone could have been filled at on demand. Everything
        downstream of this is labelled reconstructed for that reason, and none
        of it is written into the recorded-chain archive.

        Unlike get_stock_bars this tool does take a page token, verified against
        the running server rather than assumed.
        """
        if not symbols:
            return {}
        collected: dict[str, list[dict]] = {}
        for batch in _batched(list(symbols), _SYMBOLS_PER_REQUEST):
            page_token: str | None = None
            while True:
                payload: dict[str, Any] = {
                    "symbols": ",".join(batch),
                    "timeframe": "1Min",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "limit": _MAX_BARS_PER_PAGE,
                    "sort": "asc",
                }
                if page_token:
                    payload["page_token"] = page_token
                raw = self._client.call("get_option_bars", payload)
                for symbol, bars in (raw.get("bars") or {}).items():
                    collected.setdefault(symbol, []).extend(bars or [])
                page_token = raw.get("next_page_token")
                if not page_token:
                    break
        return collected

    # --------------------------------------------------------------- execution

    def submit_structure(
        self,
        legs: list[Leg],
        contracts: int,
        limit_price: float,
        client_order_id: str,
    ) -> OrderRecord:
        """Send one structure as a single atomic multi-leg order.

        Alpaca fills every leg of a multi-leg order together or none of them,
        which is what removes partial-fill risk from a four-legged structure.
        The order is a limit, never a market: a market order on a spread is a
        donation to the other side, and Law 7 exists to prevent exactly that.

        The limit is the net of the whole structure, positive for a debit and
        negative for a credit, which is how the broken-wing butterfly gets sent
        for the credit it is usually worth.
        """
        if contracts <= 0:
            raise ExecutionError(f"refusing to submit an order for {contracts} contracts")
        if not legs:
            raise ExecutionError("refusing to submit an order with no legs")
        if len(legs) > 4:
            raise ExecutionError(
                f"Alpaca accepts at most four legs in a multi-leg order, this has {len(legs)}"
            )

        payload = {
            "qty": str(contracts),
            "type": "limit",
            "time_in_force": "day",
            "order_class": "mleg",
            "limit_price": f"{limit_price:.2f}",
            "client_order_id": client_order_id,
            "legs": [
                {
                    "symbol": leg.contract.symbol,
                    "ratio_qty": str(abs(leg.ratio)),
                    "side": "buy" if leg.ratio > 0 else "sell",
                    "position_intent": "buy_to_open" if leg.ratio > 0 else "sell_to_open",
                }
                for leg in legs
            ],
        }
        return _order_record(self._client.call("place_option_order", payload))

    def close_leg(
        self,
        symbol: str,
        held_contracts: int,
        limit_price: float,
        client_order_id: str,
    ) -> OrderRecord:
        """Close one option leg with a marketable limit, never a market order.

        ``held_contracts`` is the signed position: positive is long and is sold
        to close, negative is short and is bought to close. The caller supplies
        the limit from the live touch, so the price is measured rather than
        surrendered to whatever the book happens to be showing when the order
        lands.
        """
        if held_contracts == 0:
            raise ExecutionError(f"{symbol}: refusing to close a position of zero contracts")
        if limit_price <= 0.0:
            raise ExecutionError(f"{symbol}: refusing to close at a limit of {limit_price}")

        long_position = held_contracts > 0
        return _order_record(
            self._client.call(
                "place_option_order",
                {
                    "symbol": symbol,
                    "qty": str(abs(held_contracts)),
                    "type": "limit",
                    "time_in_force": "day",
                    "side": "sell" if long_position else "buy",
                    "position_intent": "sell_to_close" if long_position else "buy_to_close",
                    "limit_price": f"{limit_price:.2f}",
                    "client_order_id": client_order_id,
                },
            )
        )

    def order(self, order_id: str) -> OrderRecord:
        return _order_record(self._client.call("get_order_by_id", {"order_id": order_id}))

    def positions(self) -> list[PositionRecord]:
        raw = self._client.call("get_all_positions")
        if not isinstance(raw, list):
            raise DataError(f"get_all_positions returned {type(raw).__name__}, not a list")
        return [
            PositionRecord(
                symbol=str(_require(row, "symbol", "get_all_positions")),
                qty=str(_require(row, "qty", "get_all_positions")),
                asset_class=str(row.get("asset_class", "")),
                avg_entry_price=str(_require(row, "avg_entry_price", "get_all_positions")),
                unrealized_pl=str(_require(row, "unrealized_pl", "get_all_positions")),
                market_value=str(_require(row, "market_value", "get_all_positions")),
            )
            for row in raw
        ]


def _order_record(raw: Any) -> OrderRecord:
    if not isinstance(raw, dict):
        raise ExecutionError(f"the order tool returned {type(raw).__name__}, not an order")
    context = "order response"
    return OrderRecord(
        id=str(_require(raw, "id", context)),
        status=str(_require(raw, "status", context)),
        submitted_at=str(raw.get("submitted_at", "")),
        client_order_id=str(raw.get("client_order_id", "")),
        filled_qty=str(raw.get("filled_qty", "0")),
        filled_avg_price=(
            str(raw["filled_avg_price"]) if raw.get("filled_avg_price") is not None else None
        ),
    )


def _greeks(snapshot: dict) -> Greeks | None:
    raw = snapshot.get("greeks")
    implied = snapshot.get("impliedVolatility")
    if not raw or implied is None:
        return None
    context = "option snapshot greeks"
    return Greeks(
        delta=_float(raw, "delta", context),
        gamma=_float(raw, "gamma", context),
        theta=_float(raw, "theta", context),
        vega=_float(raw, "vega", context),
        rho=_float(raw, "rho", context),
        implied_volatility=float(implied),
    )


def _volume(snapshot: dict) -> int | None:
    """Session volume for the contract, when the snapshot carries a daily bar."""
    bar = snapshot.get("dailyBar")
    if not bar or bar.get("v") is None:
        return None
    return int(bar["v"])
