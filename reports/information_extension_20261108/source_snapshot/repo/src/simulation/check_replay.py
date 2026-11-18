"""Compare two independent Geant4 chunks run with identical inputs and seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    a = json.loads((args.first / "manifest.json").read_text(encoding="utf-8"))
    b = json.loads((args.second / "manifest.json").read_text(encoding="utf-8"))
    required_equal = (
        "events",
        "beam_seed",
        "transport_seed",
        "thickness_mm",
        "geant4_runtime_number",
        "physics_list",
        "slab_material",
        "requested_cut_mm",
        "step_fraction_candidate",
        "beam_input_sha256",
        "run_id",
        "chunk_id",
        "full_secondary_step_policy",
    )
    for key in required_equal:
        if a.get(key) != b.get(key):
            raise ValueError(f"run settings differ: {key}")
    files = sorted(a["files_sha256"])
    if files != sorted(b["files_sha256"]):
        raise ValueError("run output file sets differ")
    differences = [name for name in files if digest(args.first / name) != digest(args.second / name)]
    report = {"passed": not differences, "different_files": differences, "compared_files": files}
    print(json.dumps(report, indent=2))
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
