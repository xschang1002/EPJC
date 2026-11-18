"""Run a fixed beam catalog through independent serial Geant4 processes.

The catalog is input-only. Each chunk is audited before it is accepted; a
restart validates existing outputs and never silently recounts an event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_script(script: str, *args: str) -> dict:
    completed = subprocess.run([sys.executable, str(Path(__file__).with_name(script)), *args], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"{script} failed: {completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout)


def validate_existing_chunk(manifest: dict, item: dict, run_id: str, *, cut_mm: float, step_fraction: float) -> None:
    """Bind a resumed chunk to both the input catalog and requested transport cuts."""
    for name, expected in (
        ("beam_input_sha256", item["input_sha256"]),
        ("transport_seed", item["transport_seed"]),
        ("events", item["events"]),
        ("run_id", run_id),
        ("chunk_id", item["chunk_id"]),
    ):
        if manifest.get(name) != expected:
            raise ValueError(f"existing chunk {item['chunk_id']} differs in {name}")
    for name, expected in (
        ("thickness_mm", item["thickness_mm"]),
        ("requested_cut_mm", cut_mm),
        ("step_fraction_candidate", step_fraction),
        ("max_slab_step_mm_requested", item["thickness_mm"] * step_fraction),
    ):
        value = manifest.get(name)
        if value is None or not math.isfinite(float(value)) or not math.isclose(float(value), float(expected), rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"existing chunk {item['chunk_id']} differs in {name}; use a new output directory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--geant4-data-dir", type=Path, required=True)
    parser.add_argument("--cut-mm", type=float, default=0.01)
    parser.add_argument("--step-fraction", type=float, default=0.1)
    parser.add_argument("--record-secondary-steps", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.cut_mm) or args.cut_mm <= 0 or not 0 < args.step_fraction <= 1:
        parser.error("cut-mm must be positive and step-fraction must be in (0, 1]")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    chunks = catalog["chunks"]
    if catalog["cells"] != 42 or len(chunks) != 42:
        raise ValueError("catalog must contain 42 fixed-stratum chunks")
    if sum(item["events"] for item in chunks) != catalog["events"]:
        raise ValueError("catalog event count mismatch")
    if len({item["transport_seed"] for item in chunks}) != len(chunks):
        raise ValueError("transport seed reused across chunks")
    if [item["chunk_id"] for item in chunks] != list(range(len(chunks))):
        raise ValueError("chunk IDs must be 0..41 in catalog order")
    args.output.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    for item in chunks:
        input_path = (args.catalog.parent / item["input_file"]).resolve(strict=True)
        if input_path.parent != args.catalog.parent.resolve(strict=True):
            raise ValueError("input file escapes catalog directory")
        if sha256(input_path) != item["input_sha256"]:
            raise ValueError(f"beam input hash mismatch in chunk {item['chunk_id']}")
        chunk_id = item["chunk_id"]
        chunk_dir = args.output / f"chunk_{chunk_id:02d}"
        log_path = args.logs / f"chunk_{chunk_id:02d}.log"
        if chunk_dir.exists():
            manifest = json.loads((chunk_dir / "manifest.json").read_text(encoding="utf-8"))
            validate_existing_chunk(
                manifest, item, catalog["run_id"], cut_mm=args.cut_mm, step_fraction=args.step_fraction
            )
            resumed = True
        else:
            if chunk_dir.with_name(chunk_dir.name + ".partial").exists():
                raise RuntimeError(f"partial chunk requires investigation: {chunk_id}")
            command = [
                sys.executable, str(Path(__file__).with_name("w0_smoke.py")),
                "--events", str(item["events"]),
                "--output", str(chunk_dir),
                "--input-events-jsonl", str(input_path),
                "--run-id", catalog["run_id"],
                "--chunk-id", str(chunk_id),
                "--transport-seed", str(item["transport_seed"]),
                "--thickness-mm", str(item["thickness_mm"]),
                "--cut-mm", str(args.cut_mm),
                "--step-fraction", str(args.step_fraction),
                "--geant4-data-dir", str(args.geant4_data_dir),
            ]
            if args.record_secondary_steps:
                command.append("--record-secondary-steps")
            with log_path.open("xb") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"Geant4 chunk {chunk_id} failed ({completed.returncode}); inspect {log_path}")
            resumed = False
        manifest = json.loads((chunk_dir / "manifest.json").read_text(encoding="utf-8"))
        audit = checked_script("audit_w0.py", str(chunk_dir))
        if args.record_secondary_steps and manifest["full_secondary_step_policy"] != "all_secondary_tracks_all_volumes":
            raise ValueError(f"chunk {chunk_id}: full secondary steps were not recorded")
        cuts = checked_script("audit_cut_log.py", str(log_path), "--expected-mm", str(args.cut_mm))
        steps = checked_script("audit_step_limit.py", str(chunk_dir))
        (args.logs / f"chunk_{chunk_id:02d}_audit.json").write_text(
            json.dumps({"ledger": audit, "cuts": cuts, "step_limit": steps}, indent=2), encoding="utf-8"
        )
        print(json.dumps({"chunk_id": chunk_id, "cell": item["cell"], "events": item["events"], "resumed": resumed}), flush=True)


if __name__ == "__main__":
    main()
