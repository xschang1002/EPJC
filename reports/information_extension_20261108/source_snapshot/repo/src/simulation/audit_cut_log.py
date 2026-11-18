"""Read back Geant4's material-couple production range cuts from a run log."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


CUT_PATTERN = re.compile(
    r"gamma\s+(?P<gamma>[0-9.eE+-]+)\s+(?P<gamma_unit>\w+)\s+"
    r"e-\s+(?P<electron>[0-9.eE+-]+)\s+(?P<electron_unit>\w+)\s+"
    r"e\+\s+(?P<positron>[0-9.eE+-]+)\s+(?P<positron_unit>\w+)\s+"
    r"proton\s+(?P<proton>[0-9.eE+-]+)\s+(?P<proton_unit>\w+)"
)
MM_PER_UNIT = {"nm": 1e-6, "um": 1e-3, "mm": 1.0, "cm": 10.0, "m": 1000.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-mm", type=float, required=True)
    parser.add_argument("--materials", nargs="+", default=["G4_Si", "G4_Galactic"])
    args = parser.parse_args()
    if args.expected_mm <= 0:
        parser.error("expected-mm must be positive")

    material = None
    found = {}
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Material :" in line:
            material = line.split("Material :", 1)[1].strip()
        elif "Range cuts" in line and material is not None:
            match = CUT_PATTERN.search(line)
            if not match:
                raise ValueError(f"unparseable Geant4 cut table line: {line}")
            values = {}
            for particle, label in (("gamma", "gamma"), ("e-", "electron"), ("e+", "positron"), ("proton", "proton")):
                unit = match[f"{label}_unit"]
                if unit not in MM_PER_UNIT:
                    raise ValueError(f"unknown range unit: {unit}")
                value = float(match[label]) * MM_PER_UNIT[unit]
                if not math.isclose(value, args.expected_mm, rel_tol=1e-8, abs_tol=1e-12):
                    raise ValueError(f"{material} {particle} cut {value} mm != {args.expected_mm} mm")
                values[particle] = value
            found[material] = values
            material = None
    if not set(args.materials).issubset(found):
        raise ValueError(f"cut table lacks required material couples: {found.keys()}")
    print(json.dumps({"passed": True, "expected_mm": args.expected_mm, "material_couples_mm": found}, indent=2))


if __name__ == "__main__":
    main()
