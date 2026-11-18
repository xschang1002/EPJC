"""Compare recorded raw Geant4 files across two 42-chunk runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--single", action="store_true", help="compare two individual chunk directories")
    parser.add_argument("--require-identical", action="store_true")
    args = parser.parse_args()
    counts = {}
    differences = []
    pairs = [(None, args.first, args.second)] if args.single else [(i, args.first / f"chunk_{i:02d}", args.second / f"chunk_{i:02d}") for i in range(42)]
    for i, first_root, second_root in pairs:
        first = json.loads((first_root / "manifest.json").read_text(encoding="utf-8"))
        second = json.loads((second_root / "manifest.json").read_text(encoding="utf-8"))
        for name in sorted(set(first["files_sha256"]) & set(second["files_sha256"])):
            counts[name] = counts.get(name, 0) + (first["files_sha256"][name] == second["files_sha256"][name])
            if first["files_sha256"][name] != second["files_sha256"][name]:
                differences.append({"chunk_id": i, "file": name})
    result = {"compared_chunks": len(pairs), "same_file_counts": counts, "differences": differences}
    print(json.dumps(result, indent=2))
    if args.require_identical and differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
