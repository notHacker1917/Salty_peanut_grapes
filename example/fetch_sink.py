#!/usr/bin/env python3
"""Fetch a sink from the Cula API and print it as JSON (usage example)."""

from __future__ import annotations

import argparse
import sys
from uuid import UUID

from cula import CulaClient


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a sink by UUID, or the first listed sink if omitted."
    )
    parser.add_argument(
        "sink_id",
        nargs="?",
        metavar="UUID",
        help="Sink id (default: first id from GET /sinks)",
    )
    args = parser.parse_args()

    with CulaClient() as client:
        if args.sink_id:
            try:
                sink_id = UUID(args.sink_id)
            except ValueError:
                print(f"Invalid UUID: {args.sink_id!r}", file=sys.stderr)
                return 1
            sink = client.get_sink(sink_id)
        else:
            ids = client.list_sinks()
            if not ids:
                print("No sinks returned by the API.", file=sys.stderr)
                return 1
            sink = client.get_sink(ids[0])

    print(sink.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
