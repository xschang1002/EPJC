"""Verify an archival tree against its copied tree by SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("backup", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    backup = args.backup.resolve(strict=True)
    src = {p.relative_to(source).as_posix(): p for p in source.rglob("*") if p.is_file()}
    dst = {p.relative_to(backup).as_posix(): p for p in backup.rglob("*") if p.is_file()}
    missing = sorted(set(src) - set(dst))
    extra = sorted(set(dst) - set(src))
    mismatches = []
    for rel in sorted(set(src) & set(dst)):
        if src[rel].stat().st_size != dst[rel].stat().st_size or digest(src[rel]) != digest(dst[rel]):
            mismatches.append(rel)
    result = {
        "passed": not missing and not extra and not mismatches,
        "source": str(source),
        "backup": str(backup),
        "source_files": len(src),
        "backup_files": len(dst),
        "missing": missing,
        "extra": extra,
        "mismatches": mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "passed": result["passed"], "files": len(src)}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
