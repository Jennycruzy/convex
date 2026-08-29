"""A synchronous client for Alpaca's MCP server.

The hackathon's theme is that the agent reaches the market through the Model
Context Protocol, so every chain, quote, Greek, calendar lookup and order in
this project goes through the server Alpaca publishes rather than through the
SDK directly. That server is asynchronous and speaks JSON-RPC over a pipe; the
rest of CONVEX is synchronous, deterministic and single-threaded on purpose.
This module is the seam between the two.

It owns one long-lived subprocess and one asyncio loop on a background thread.
Calls from the decision cycle are handed to that loop and waited on, so the
decision code never becomes async and never grows an await in the middle of a
risk calculation.

Two failure modes are handled explicitly because they are the ones that would
otherwise be silent:

  a tool that reports an error in its payload   The order tools answer with an
  ``{"error": ...}`` body and a successful JSON-RPC result. Nothing about the
  transport says anything went wrong, so the payload is inspected and a
  rejected order raises here instead of being read as a fill.

  a call that never returns   Every call carries a timeout. A decision cycle
  that hangs at 10:00 is a decision cycle that misses its entry.
"""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from convex.errors import ConvexError, DataError, ExecutionError


class McpError(ConvexError):
    """The MCP server could not be reached, or answered with an error."""


@dataclass(frozen=True)
class McpSettings:
    """How to start the server and how long to wait for it."""

    command: str
    args: tuple[str, ...]
    env: dict[str, str]
    log_path: Path | None = None
    startup_timeout: float = 60.0
    call_timeout: float = 45.0


def _unwrap(payload: Any) -> Any:
    """FastMCP wraps a non-object return in ``{"result": ...}``; undo that."""
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


SECURITY_KEY = "_alpaca_mcp_security"


def _strip_security_envelope(payload: Any, tool: str) -> Any:
    """Return the body of the server's provenance envelope.

    The Alpaca server does not answer with the account, the chain or the order
    directly. It answers with

        {"_alpaca_mcp_security": {...}, "data": {...}}

    where the marker declares the result untrusted tool output and asks that it
    be read as data rather than obeyed as instructions. That is exactly how this
    program treats it: nothing downstream interprets a field as an instruction,
    and the language model is handed figures that deterministic code has already
    computed. So the marker is stripped here and the body returned.

    An envelope without a body is not a shape to guess at, so it raises.
    """
    if not isinstance(payload, dict) or SECURITY_KEY not in payload:
        return payload
    if "data" not in payload:
        raise DataError(
            f"{tool} returned a {SECURITY_KEY} envelope with no 'data' body "
            f"(keys: {sorted(payload)})"
        )
    return payload["data"]


class McpClient:
    """One live connection to the Alpaca MCP server."""

    def __init__(self, settings: McpSettings) -> None:
        self._settings = settings
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self.tool_names: frozenset[str] = frozenset()

    # ----------------------------------------------------------------- lifetime

    def start(self) -> "McpClient":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run_loop, name="convex-mcp", daemon=True)
        self._thread.start()
        if not self._ready.wait(self._settings.startup_timeout):
            raise McpError(
                f"the Alpaca MCP server did not come up within "
                f"{self._settings.startup_timeout:.0f}s ({self._settings.command})"
            )
        if self._startup_error is not None:
            raise McpError(
                f"the Alpaca MCP server failed to start: {self._startup_error}"
            ) from self._startup_error
        return self

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        finally:
            loop.close()

    async def _serve(self) -> None:
        stop = asyncio.Event()
        self._stop = stop
        try:
            async with AsyncExitStack() as stack:
                parameters = StdioServerParameters(
                    command=self._settings.command,
                    args=list(self._settings.args),
                    env=dict(self._settings.env),
                )
                # The server writes a banner and its own tracebacks to stderr.
                # They are kept, not discarded: a stack trace from the market
                # side of the connection is exactly what is wanted after a bad
                # cycle, and it does not belong interleaved with the decision
                # output on the console.
                errlog = None
                if self._settings.log_path is not None:
                    self._settings.log_path.parent.mkdir(parents=True, exist_ok=True)
                    errlog = stack.enter_context(self._settings.log_path.open("a"))
                read, write = await stack.enter_async_context(
                    stdio_client(parameters, errlog=errlog) if errlog is not None
                    else stdio_client(parameters)
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listing = await session.list_tools()
                self.tool_names = frozenset(tool.name for tool in listing.tools)
                self._session = session
                self._stack = stack
                self._ready.set()
                await stop.wait()
        except BaseException as error:  # recorded, then re-raised to the caller
            self._startup_error = error
            self._ready.set()
            raise

    def require_tools(self, *names: str) -> None:
        """Fail at startup if the server does not expose what CONVEX needs.

        This runs once, before any decision is made, so a server version that
        has dropped or renamed a tool is a startup failure rather than a
        missing chain discovered halfway through a cycle at 10:00.
        """
        missing = sorted(set(names) - self.tool_names)
        if missing:
            raise McpError(
                f"the Alpaca MCP server does not expose {', '.join(missing)}; "
                f"it offers {len(self.tool_names)} tools"
            )

    def close(self) -> None:
        """Ask the server to shut down and wait for the subprocess to go.

        The loop is not stopped from underneath its own coroutine: the serve
        task is asked to finish, so the exit stack unwinds, the pipes close and
        the subprocess is reaped. Killing the loop instead leaves an orphaned
        server holding an open connection to the paper account.
        """
        loop, thread, stop = self._loop, self._thread, self._stop
        if loop is None or thread is None:
            return
        if stop is not None:
            loop.call_soon_threadsafe(stop.set)
        thread.join(timeout=10.0)
        if thread.is_alive():
            raise McpError("the MCP server did not shut down within 10s")
        self._loop = self._thread = self._session = self._stop = None

    def __enter__(self) -> "McpClient":
        return self.start()

    def __exit__(self, *_) -> None:
        self.close()

    # --------------------------------------------------------------------- calls

    def call(self, tool: str, arguments: dict[str, Any] | None = None) -> Any:
        """Invoke one tool and return its decoded payload."""
        if self._session is None or self._loop is None:
            raise McpError(f"cannot call {tool}: the MCP client has not been started")

        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(tool, arguments or {}), self._loop
        )
        try:
            result = future.result(timeout=self._settings.call_timeout)
        except FutureTimeoutError as error:
            future.cancel()
            raise McpError(
                f"{tool} did not answer within {self._settings.call_timeout:.0f}s"
            ) from error
        except Exception as error:
            raise McpError(f"{tool} failed: {error}") from error

        if getattr(result, "isError", False):
            raise McpError(f"{tool} returned an error: {_text_of(result)}")

        payload = _unwrap(getattr(result, "structuredContent", None))
        if payload is None:
            text = _text_of(result)
            if not text:
                raise DataError(f"{tool} returned an empty payload")
            try:
                payload = _unwrap(json.loads(text))
            except json.JSONDecodeError as error:
                raise DataError(f"{tool} returned something that is not JSON: {text[:200]}") from error

        # Strip the provenance envelope before anything reads the payload, so
        # that the error check below inspects the body rather than the marker.
        # The FastMCP ``{"result": ...}`` wrapper sits inside that body, not
        # outside it, so a list-returning tool needs unwrapping a second time.
        payload = _unwrap(_strip_security_envelope(payload, tool))

        # The order tools answer with a successful result whose body carries an
        # error object. Reading that as a fill is the worst failure available
        # to this program, so it is checked on every call, not just on orders.
        if isinstance(payload, dict) and "error" in payload:
            detail = payload["error"]
            message = detail.get("message") if isinstance(detail, dict) else str(detail)
            if tool.startswith("place_"):
                raise ExecutionError(f"{tool} was rejected: {message}")
            raise DataError(f"{tool} reported an error: {message}")
        return payload


def _text_of(result: Any) -> str:
    """Join whatever text blocks a tool result carries."""
    blocks = getattr(result, "content", None) or []
    return "".join(getattr(block, "text", "") for block in blocks).strip()
