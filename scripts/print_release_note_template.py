#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.version import PRODUCT_VERSION, release_note_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a structured release-note template aligned with model governance fields."
    )
    parser.add_argument(
        "--version",
        default=PRODUCT_VERSION,
        help="Release version heading to scaffold.",
    )
    parser.add_argument(
        "--date",
        default=str(date.today()),
        help="Release date in YYYY-MM-DD.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(release_note_template(args.version, args.date))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
