"""Reconcile older CONVEX order receipts against broker fills.

By default this is read-only and prints what would be appended. Pass --write
only after reviewing the broker-derived figures. It never sends, cancels, or
modifies an order and it never rewrites the JSONL ledger.

Run: .venv/bin/python -m scripts.reconcile
     .venv/bin/python -m scripts.reconcile --write
"""

from __future__ import annotations

import argparse
import sys

from convex.config import load
from convex.data.alpaca import AlpacaGateway
from convex.errors import ConvexError
from convex.ledger import Ledger
from convex.reconciliation import reconcile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="append validated broker reconciliation receipts to the ledger",
    )
    arguments = parser.parse_args()

    config = load()
    gateway = AlpacaGateway(config)
    ledger = Ledger(config.path_("paths.ledger"))
    results = reconcile(gateway, ledger, write=arguments.write)
    if not results:
        print("no unreconciled broker-filled structures found")
        return 0

    total = 0.0
    for result in results:
        total += result["realised_pnl"]
        print(
            f"{result['structure']:<16} entry {result['entry_order_id']}  "
            f"gross {result['gross_realised_pnl']:+.2f}  fees {result['broker_fees']:.2f}  "
            f"net {result['realised_pnl']:+.2f}"
        )
    mode = "appended" if arguments.write else "not written (pass --write to append)"
    print(f"portfolio {total:+.2f}; {mode}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConvexError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
