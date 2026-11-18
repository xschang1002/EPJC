"""Guarded single-block confirmation runner.

The confirmation guard is called before creating any confirmation input or
output path. This wrapper only becomes executable after a separately approved
environment record enables confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def confirmation_guard_path() -> Path:
    """The handoff layout keeps protocol/ beside src/, not inside it."""
    return Path(__file__).resolve().parents[2] / "protocol" / "confirmation_guard.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--geant4-data-dir", type=Path, required=True)
    parser.add_argument("--cut-mm", type=float, default=0.01)
    parser.add_argument("--step-fraction", type=float, default=0.1)
    parser.add_argument("--record-secondary-steps", action="store_true")
    args = parser.parse_args()
    guard = confirmation_guard_path()
    checked = subprocess.run(
        [sys.executable, str(guard), "--environment", str(args.environment)],
        capture_output=True,
        text=True,
    )
    print(checked.stdout, end="")
    if checked.returncode:
        if checked.stderr:
            print(checked.stderr, file=sys.stderr, end="")
        return checked.returncode
    # Only after the guard allows it do we create or touch confirmation paths.
    run_chunks = Path(__file__).with_name("run_chunks.py")
    merge = Path(__file__).resolve().parents[1] / "data_contract" / "merge_chunks.py"
    command = [
        sys.executable,
        str(run_chunks),
        str(args.catalog),
        "--output", str(args.output),
        "--logs", str(args.logs),
        "--geant4-data-dir", str(args.geant4_data_dir),
        "--cut-mm", str(args.cut_mm),
        "--step-fraction", str(args.step_fraction),
    ]
    if args.record_secondary_steps:
        command.append("--record-secondary-steps")
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(merge), str(args.catalog), str(args.output), "--output", str(args.merged)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
