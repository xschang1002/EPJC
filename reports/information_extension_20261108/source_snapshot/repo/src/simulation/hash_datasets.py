"""Hash an installed Geant4 physics dataset tree for provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    datasets = {}
    tree_digest = hashlib.sha256()
    for dataset in sorted(path for path in root.iterdir() if path.is_dir()):
        dataset_digest = hashlib.sha256()
        count = 0
        size = 0
        for path in sorted(path for path in dataset.rglob("*") if path.is_file()):
            if path.is_symlink():
                raise ValueError(f"dataset contains symlink: {path}")
            relative = path.relative_to(root).as_posix()
            record = f"{relative}\0{file_sha256(path)}\n".encode("utf-8")
            dataset_digest.update(record)
            tree_digest.update(record)
            count += 1
            size += path.stat().st_size
        datasets[dataset.name] = {
            "files": count,
            "bytes": size,
            "sha256_tree": dataset_digest.hexdigest(),
        }
    if not datasets or any(item["files"] == 0 for item in datasets.values()):
        raise ValueError("dataset tree is empty or contains an empty dataset directory")
    manifest = {
        "source_dir": str(root),
        "algorithm": "SHA256(sorted UTF-8 relative path, NUL, file SHA256, newline)",
        "datasets": datasets,
        "sha256_tree": tree_digest.hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "datasets": list(datasets), "tree": manifest["sha256_tree"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
