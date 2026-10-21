"""Check recorded public-source hashes without executing research code."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Optional local original source root")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    with (repo / "docs/source_file_mapping.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    failures = []
    seen = set()
    for row in rows:
        relative = row["destination_path"]
        if row["verification_status"] not in {"verified_identical", "verified_publication_edit"}:
            failures.append(f"Unknown verification status: {relative}")
        if relative.casefold() in seen:
            failures.append(f"Duplicate destination: {relative}")
        seen.add(relative.casefold())
        try:
            path = contained(repo, relative)
            actual = digest(path)
            if actual != row["destination_sha256"]:
                failures.append(f"Changed or corrupted destination: {relative}")
            if row["verification_status"] == "verified_identical" and actual != row["source_sha256"]:
                failures.append(f"Original-source hash mismatch: {relative}")
            if args.source_root:
                source = contained(args.source_root, row["source_path"])
                if digest(source) != row["source_sha256"]:
                    failures.append(f"Original source changed: {row['source_path']}")
        except (OSError, ValueError) as error:
            failures.append(f"Cannot verify {relative}: {type(error).__name__}")
    if not rows:
        failures.append("The source mapping is empty")
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1
    edited = sum(row["verification_status"] == "verified_publication_edit" for row in rows)
    print(f"PASS: {len(rows)} public source assets match the recorded destination hashes.")
    print(f"PASS: {edited} documented publication edits; {len(rows) - edited} unchanged source assets.")
    if args.source_root:
        print("PASS: all supplied original-source hashes also match.")
    print("This verifies recorded file integrity; it does not validate model predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
