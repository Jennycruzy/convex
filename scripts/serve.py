"""Serve the dashboard.

Judges load a URL, not a repository, so this has to be reachable from outside
the box it runs on. It binds every interface by default for that reason.

Run it with:  .venv/bin/python -m scripts.serve --port 8000
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from convex.config import load
from convex.dashboard.app import create_app
from convex.errors import ConvexError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="port to bind")
    parser.add_argument("--reload", action="store_true", help="reload on source changes")
    arguments = parser.parse_args()

    config = load()
    ledger = config.path_("paths.ledger")
    print(f"reading the ledger at {ledger}")
    if not ledger.is_file():
        print("  it does not exist yet; the page will say so rather than inventing data")

    uvicorn.run(
        "scripts.serve:app" if arguments.reload else create_app(config),
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
        log_level="info",
    )
    return 0


app = create_app(load())


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
