"""Create deterministic fixed-thickness, fixed-stratum input chunks.

These files describe incident pions only. No physics output is synthesized.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_contract.beam_schedule import CELLS, chunk_transport_seeds, event_spec


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_name(bin_index: int, pdg: int, thickness: float) -> str:
    charge = "plus" if pdg == 211 else "minus"
    return f"pbin{bin_index}_{charge}_t{str(thickness).replace('.', 'p')}mm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--events", type=int, default=10000)
    parser.add_argument("--beam-seed", type=int, default=20261010)
    parser.add_argument("--transport-master-seed", type=int, default=314159)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.events < len(CELLS):
        parser.error("events must cover all 42 cells")
    if not args.run_id or ":" in args.run_id:
        parser.error("run-id must be nonempty and cannot contain ':'")
    pending = args.output.with_name(args.output.name + ".partial")
    if args.output.exists() or pending.exists():
        parser.error("output or partial directory already exists")
    pending.mkdir(parents=True)
    cell_to_chunk = {cell: i for i, cell in enumerate(CELLS)}
    seeds = chunk_transport_seeds(args.transport_master_seed, list(range(len(CELLS))))
    paths = [pending / f"chunk_{i:02d}.jsonl" for i in range(len(CELLS))]
    counts = [0] * len(CELLS)
    with ExitStack() as stack:
        streams = [stack.enter_context(path.open("w", encoding="utf-8", newline="\n")) for path in paths]
        for index in range(args.events):
            event = event_spec(args.run_id, index, args.beam_seed)
            chunk_id = cell_to_chunk[(event.momentum_bin, event.pdg, event.thickness_mm)]
            streams[chunk_id].write(json.dumps(event.as_record(), sort_keys=True) + "\n")
            counts[chunk_id] += 1
    if sum(counts) != args.events or any(count == 0 for count in counts):
        raise RuntimeError("beam allocation failed to cover exactly all requested events")
    chunks = []
    for chunk_id, (cell, path, count) in enumerate(zip(CELLS, paths, counts)):
        bin_index, pdg, thickness = cell
        chunks.append({
            "chunk_id": chunk_id,
            "cell": cell_name(bin_index, pdg, thickness),
            "momentum_bin": bin_index,
            "pdg": pdg,
            "thickness_mm": thickness,
            "events": count,
            "transport_seed": seeds[chunk_id],
            "input_file": path.name,
            "input_sha256": sha256(path),
        })
    catalog = {
        "stage": "W1_BEAM_INPUT_ONLY_NO_PHYSICS",
        "run_id": args.run_id,
        "events": args.events,
        "beam_seed": args.beam_seed,
        "transport_master_seed": args.transport_master_seed,
        "cells": len(CELLS),
        "chunks": chunks,
    }
    (pending / "catalog.json").write_text(json.dumps(catalog, indent=2, sort_keys=True), encoding="utf-8")
    pending.rename(args.output)
    print(json.dumps({"output": str(args.output), "events": args.events, "chunks": len(chunks)}))


if __name__ == "__main__":
    main()
