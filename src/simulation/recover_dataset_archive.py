"""Recover an interrupted official Geant4 archive with a checked HTTP range.

The source partial is read only. The completed archive is written under the
project data directory and never used until its hash and tar structure pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-md5")
    args = parser.parse_args()
    present = args.partial.stat().st_size
    if not 0 < present < args.expected_size:
        raise ValueError("partial file size is outside the expected range")
    if args.output.exists():
        raise FileExistsError(args.output)
    requested = f"bytes={present}-{args.expected_size - 1}"
    with urlopen(Request(args.url, headers={"Range": requested}), timeout=120) as response:
        content_range = response.headers.get("Content-Range", "")
        expected_range = f"bytes {present}-{args.expected_size - 1}/{args.expected_size}"
        if response.status != 206 or content_range != expected_range:
            raise RuntimeError(f"server refused requested range: {response.status} {content_range!r}")
        remainder = response.read()
    if len(remainder) != args.expected_size - present:
        raise ValueError(f"incomplete range: got {len(remainder)} expected {args.expected_size - present}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.name + ".partial")
    if pending.exists():
        raise FileExistsError(pending)
    digest = hashlib.sha256()
    md5_digest = hashlib.md5(usedforsecurity=False)
    with args.partial.open("rb") as source, pending.open("xb") as destination:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            destination.write(block)
            digest.update(block)
            md5_digest.update(block)
        destination.write(remainder)
        digest.update(remainder)
        md5_digest.update(remainder)
    actual = digest.hexdigest()
    if pending.stat().st_size != args.expected_size:
        raise ValueError("completed archive size mismatch")
    if args.expected_sha256 and actual.lower() != args.expected_sha256.lower():
        raise ValueError(f"archive SHA-256 mismatch: {actual}")
    if args.expected_md5 and md5_digest.hexdigest().lower() != args.expected_md5.lower():
        raise ValueError(f"archive official MD5 mismatch: {md5_digest.hexdigest()}")
    with tarfile.open(pending, "r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("archive has no members")
        roots = {Path(member.name).parts[0] for member in members if Path(member.name).parts}
        if len(roots) != 1:
            raise ValueError(f"unexpected archive roots: {roots}")
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError(f"unsafe archive entry: {member.name}")
    pending.rename(args.output)
    print(json.dumps({"archive": str(args.output), "sha256": actual, "md5": md5_digest.hexdigest(), "bytes": args.expected_size, "root": next(iter(roots)), "members": len(members)}))


if __name__ == "__main__":
    main()
