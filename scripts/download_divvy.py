#!/usr/bin/env python3
"""Download the official 2024 Divvy archives and verify their checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bwar.paper_jcgs.divvy_data import divvy_zip_path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "divvy_source_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the official Divvy source archives."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download missing files.",
    )
    args = parser.parse_args()

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    for source in payload["sources"]:
        month = source["month"]
        expected = source["sha256"]
        path = ROOT / source["path"]
        if not path.exists() and not args.verify_only:
            path = divvy_zip_path(month)
        if not path.exists():
            failures.append(f"{month}: missing {path}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(
                f"{month}: checksum mismatch ({actual}; expected {expected})"
            )
        else:
            print(f"{month}: verified")

    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
