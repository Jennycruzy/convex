"""The Alpaca data and execution layer.

Every market fact the agent uses enters here: the account, the calendar, the
0DTE chain with Greeks, implied volatility and open interest, the underlying's
history, and the multi-leg orders that go back out. There is no other source,
no second vendor, and no path that returns a fabricated value when a call
fails. A missing credential, an unentitled feed, an expiry that does not exist,
a contract with no quote: each raises.

Chain assembly needs both Alpaca APIs. The trading API lists the contracts and
carries strike, expiry, multiplier and open interest; the market data API
carries the quote, the Greeks and the implied volatility. A row is only
returned when both halves are present, because a strike the agent can see but
cannot price is not a strike it can trade.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import (
    OptionSnapshotRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    ContractType,
    OrderClass,
    OrderSide,
    PositionIntent,
    TimeInForce,
)
from alpaca.trading.requests import (
    GetCalendarRequest,
    GetOptionContractsRequest,
    LimitOrderRequest,
    OptionLegRequest,
)
from dotenv import load_dotenv

from convex.config import Config
from convex.errors import CredentialsError, DataError, ExecutionError
from convex.instruments import ChainEntry, Greeks, Leg, OptionContract, Quote, Right

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


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


class AlpacaGateway:
    """A single connection to the paper account and its market data."""

    def __init__(self, config: Config, env_path: Path | None = None) -> None:
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

        self.config = config
        self.trading = TradingClient(key, secret, paper=True)
        self.option_data = OptionHistoricalDataClient(key, secret)
        self.stock_data = StockHistoricalDataClient(key, secret)
        self._options_feed = config.str_("data.options_feed")
        self._stock_feed = config.str_("data.stock_feed")

    # ------------------------------------------------------------------ account

    def account(self) -> AccountSnapshot:
        raw = self.trading.get_account()
        return AccountSnapshot(
            account_number=str(raw.account_number),
            status=str(raw.status),
            equity=_number(raw.equity, "equity"),
            last_equity=_number(raw.last_equity, "last_equity"),
            cash=_number(raw.cash, "cash"),
            buying_power=_number(raw.buying_power, "buying_power"),
            options_buying_power=_number(raw.options_buying_power, "options_buying_power"),
            options_approved_level=int(raw.options_approved_level or 0),
            multiplier=_number(raw.multiplier, "multiplier"),
            pattern_day_trader=bool(raw.pattern_day_trader),
        )

    def clock(self) -> tuple[datetime, bool]:
        """Exchange time and whether the market is open, from Alpaca not locally."""
        raw = self.trading.get_clock()
        return raw.timestamp, bool(raw.is_open)

    def sessions(self, start: date, end: date) -> list[MarketSession]:
        """Trading sessions in the window. A closed day simply does not appear."""
        raw = self.trading.get_calendar(GetCalendarRequest(start=start, end=end))
        return [
            MarketSession(session_date=day.date, open_at=day.open, close_at=day.close)
            for day in raw
        ]

    def is_trading_day(self, day: date) -> bool:
        return any(session.session_date == day for session in self.sessions(day, day))

    # ------------------------------------------------------------- underlying

    def spot(self, symbol: str) -> tuple[float, datetime]:
        """Mid of the underlying's latest quote, with the timestamp it carries."""
        response = self.stock_data.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self._stock_feed)
        )
        if symbol not in response:
            raise DataError(f"no latest quote returned for {symbol}")
        quote = response[symbol]
        priced = Quote(
            symbol=symbol,
            bid=float(quote.bid_price),
            ask=float(quote.ask_price),
            bid_size=int(quote.bid_size),
            ask_size=int(quote.ask_size),
            timestamp=quote.timestamp,
        )
        return priced.mid, priced.timestamp

    def minute_bars(self, symbol: str, start: datetime, end: datetime):
        """Minute bars for the underlying, used to build the return history."""
        response = self.stock_data.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
                timeframe=TimeFrame.Minute,
                feed=self._stock_feed,
            )
        )
        frame = response.df
        if frame.empty:
            raise DataError(
                f"Alpaca returned no minute bars for {symbol} between {start} and {end}"
            )
        return frame

    # ------------------------------------------------------------------ chains

    def expirations(self, symbol: str, on_or_before: date) -> list[date]:
        """Expiries Alpaca actually lists for this underlying, soonest first."""
        request = GetOptionContractsRequest(
            underlying_symbols=[symbol],
            expiration_date_gte=on_or_before,
            expiration_date_lte=on_or_before + timedelta(days=14),
            limit=10_000,
        )
        contracts = self.trading.get_option_contracts(request).option_contracts
        if not contracts:
            raise DataError(f"Alpaca lists no {symbol} contracts expiring near {on_or_before}")
        return sorted({contract.expiration_date for contract in contracts})

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
            if snapshot is None or snapshot.latest_quote is None:
                continue
            quote = snapshot.latest_quote
            if float(quote.bid_price) <= 0.0 and float(quote.ask_price) <= 0.0:
                continue
            rows.append(
                ChainEntry(
                    contract=contract,
                    quote=Quote(
                        symbol=occ_symbol,
                        bid=float(quote.bid_price),
                        ask=float(quote.ask_price),
                        bid_size=int(quote.bid_size),
                        ask_size=int(quote.ask_size),
                        timestamp=quote.timestamp,
                    ),
                    greeks=_greeks(snapshot),
                    open_interest=open_interest,
                    volume=None,
                )
            )
        if not rows:
            raise DataError(
                f"no priceable {symbol} contracts for {expiry} between "
                f"{strike_low} and {strike_high} on the {self._options_feed} feed"
            )
        return sorted(rows, key=lambda row: (row.contract.right, row.contract.strike))

    def _list_contracts(
        self, symbol: str, expiry: date, strike_low: float, strike_high: float
    ) -> dict[str, tuple[OptionContract, int | None]]:
        """Tradable contracts in the band, each with the open interest Alpaca
        publishes for it. Open interest is a trading-API field, not a market-data
        one, which is why the chain is assembled from both."""
        contracts: dict[str, tuple[OptionContract, int | None]] = {}
        page_token: str | None = None
        while True:
            response = self.trading.get_option_contracts(
                GetOptionContractsRequest(
                    underlying_symbols=[symbol],
                    expiration_date=expiry,
                    strike_price_gte=str(strike_low),
                    strike_price_lte=str(strike_high),
                    limit=10_000,
                    page_token=page_token,
                )
            )
            for raw in response.option_contracts or []:
                if not raw.tradable:
                    continue
                contracts[raw.symbol] = (
                    OptionContract(
                        symbol=raw.symbol,
                        underlying=raw.underlying_symbol,
                        right=Right.CALL if raw.type == ContractType.CALL else Right.PUT,
                        strike=float(raw.strike_price),
                        expiry=raw.expiration_date,
                        multiplier=int(raw.size),
                    ),
                    int(raw.open_interest) if raw.open_interest is not None else None,
                )
            page_token = response.next_page_token
            if not page_token:
                break
        if not contracts:
            raise DataError(
                f"Alpaca lists no tradable {symbol} contracts expiring {expiry} "
                f"between strikes {strike_low} and {strike_high}"
            )
        return contracts

    def _snapshots(self, symbols: list[str]) -> dict:
        snapshots: dict = {}
        batch_size = 100
        for start in range(0, len(symbols), batch_size):
            batch = symbols[start : start + batch_size]
            snapshots.update(
                self.option_data.get_option_snapshot(
                    OptionSnapshotRequest(symbol_or_symbols=batch, feed=self._options_feed)
                )
            )
        return snapshots

    # --------------------------------------------------------------- execution

    def submit_structure(
        self,
        legs: list[Leg],
        contracts: int,
        limit_price: float,
        client_order_id: str,
    ):
        """Send one structure as a single atomic multi-leg order.

        Alpaca fills every leg of a multi-leg order together or none of them,
        which is what removes partial-fill risk from a four-legged structure.
        The order is a limit, never a market: a market order on a spread is a
        donation to the other side, and Law 7 exists to prevent exactly that.
        """
        if contracts <= 0:
            raise ExecutionError(f"refusing to submit an order for {contracts} contracts")
        if not legs:
            raise ExecutionError("refusing to submit an order with no legs")

        order_legs = [
            OptionLegRequest(
                symbol=leg.contract.symbol,
                ratio_qty=abs(leg.ratio),
                side=OrderSide.BUY if leg.ratio > 0 else OrderSide.SELL,
                position_intent=(
                    PositionIntent.BUY_TO_OPEN if leg.ratio > 0 else PositionIntent.SELL_TO_OPEN
                ),
            )
            for leg in legs
        ]
        request = LimitOrderRequest(
            qty=contracts,
            limit_price=round(limit_price, 2),
            order_class=OrderClass.MLEG,
            time_in_force=TimeInForce.DAY,
            legs=order_legs,
            client_order_id=client_order_id,
        )
        try:
            return self.trading.submit_order(request)
        except Exception as exc:  # re-raised, never swallowed
            raise ExecutionError(f"Alpaca rejected the multi-leg order: {exc}") from exc

    def close_leg(
        self,
        symbol: str,
        held_contracts: int,
        limit_price: float,
        client_order_id: str,
    ):
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
        request = LimitOrderRequest(
            symbol=symbol,
            qty=abs(held_contracts),
            limit_price=round(limit_price, 2),
            side=OrderSide.SELL if long_position else OrderSide.BUY,
            position_intent=(
                PositionIntent.SELL_TO_CLOSE if long_position else PositionIntent.BUY_TO_CLOSE
            ),
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        try:
            return self.trading.submit_order(request)
        except Exception as exc:  # re-raised, never swallowed
            raise ExecutionError(f"Alpaca rejected the closing order for {symbol}: {exc}") from exc

    def order(self, order_id: str):
        return self.trading.get_order_by_id(order_id)

    def positions(self):
        return self.trading.get_all_positions()

    def option_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        """Live two-sided quotes for contracts already held.

        The chain builder cannot serve this: it filters to a strike band around
        spot, and a leg the guard needs to close may have moved outside that
        band, which is precisely the case where closing it matters most.
        """
        if not symbols:
            return {}
        quotes: dict[str, Quote] = {}
        for snapshot_symbol, snapshot in self._snapshots(list(symbols)).items():
            if snapshot is None or snapshot.latest_quote is None:
                continue
            raw = snapshot.latest_quote
            quotes[snapshot_symbol] = Quote(
                symbol=snapshot_symbol,
                bid=float(raw.bid_price),
                ask=float(raw.ask_price),
                bid_size=int(raw.bid_size),
                ask_size=int(raw.ask_size),
                timestamp=raw.timestamp,
            )
        missing = set(symbols) - set(quotes)
        if missing:
            raise DataError(f"no quote came back for {', '.join(sorted(missing))}")
        return quotes


def _number(value, field: str) -> float:
    if value is None:
        raise DataError(f"account field {field!r} came back empty")
    return float(value)


def _greeks(snapshot) -> Greeks | None:
    raw = snapshot.greeks
    if raw is None or snapshot.implied_volatility is None:
        return None
    return Greeks(
        delta=float(raw.delta),
        gamma=float(raw.gamma),
        theta=float(raw.theta),
        vega=float(raw.vega),
        rho=float(raw.rho),
        implied_volatility=float(snapshot.implied_volatility),
    )
