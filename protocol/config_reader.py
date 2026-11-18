"""Safely parse and validate the research protocol draft.

The YAML in ``planning/pion_material_research_plan`` is a design input.  This
reader produces a canonical JSON artifact with a source hash and deliberately
preserves the draft status; it never upgrades a draft into a frozen protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(document: dict[str, Any]) -> None:
    required = {"schema_version", "status", "governance", "scope", "environment", "simulation", "endpoint", "models"}
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"protocol draft missing sections: {', '.join(missing)}")
    if document["status"] != "DRAFT_NOT_EXECUTED":
        raise ValueError("reader only accepts the supplied DRAFT_NOT_EXECUTED protocol")
    if document["governance"].get("confirmation_enabled") is not False:
        raise ValueError("confirmation must remain disabled in the draft")
    if "ACTS" not in document["scope"].get("excluded_dependencies", []):
        raise ValueError("ACTS must remain explicitly excluded")
    simulation = document["simulation"]
    if simulation.get("material_candidate") != "G4_Si":
        raise ValueError("unexpected material candidate")
    if simulation.get("physics_list_candidate") != "FTFP_BERT":
        raise ValueError("unexpected physics list candidate")
    momenta = simulation.get("momentum_GeV_c", {})
    if not (momenta.get("min") < momenta.get("max") and momenta.get("bins") == 7):
        raise ValueError("unexpected momentum grid")
    if simulation.get("thickness_mm") != [0.1, 0.3, 1.0]:
        raise ValueError("unexpected thickness grid")
    if document["environment"].get("binding_version_candidate") != "0.1.3":
        raise ValueError("unexpected binding candidate")


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("protocol YAML must contain a mapping")
    validate_protocol(document)
    return {
        "reader": "protocol.config_reader",
        "reader_status": "DRAFT_PARSED_NOT_FROZEN",
        "source_path": str(path.resolve()),
        "source_sha256": sha256(path),
        "protocol": document,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = load_protocol(args.input)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "source_sha256": result["source_sha256"], "status": result["reader_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
