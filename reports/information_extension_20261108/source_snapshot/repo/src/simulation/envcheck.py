"""Write an evidence-based W0 Geant4 environment report.

Risky native probes run in child processes, so a Geant4 fatal error remains
visible in the report instead of terminating the report writer itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import time


IMPORT_PROBE = r"""
import json
import geant4_pybind as g
classes = ('G4RunManagerFactory', 'G4VUserDetectorConstruction',
           'G4VUserPrimaryGeneratorAction', 'G4UserSteppingAction',
           'G4UserEventAction', 'G4UserTrackingAction', 'G4ParticleGun',
           'FTFP_BERT', 'G4NistManager', 'G4Step', 'G4Track', 'G4Random')
methods = {
    'G4Step': ('GetPreStepPoint', 'GetPostStepPoint',
               'GetSecondaryInCurrentStep', 'GetTrack'),
    'G4StepPoint': ('GetMomentum', 'GetKineticEnergy',
                    'GetProcessDefinedStep', 'GetTouchable'),
    'G4Track': ('GetTrackID', 'GetParentID', 'GetDefinition', 'GetTrackStatus'),
    'G4Random': ('setTheSeed', 'saveEngineStatus', 'restoreEngineStatus'),
}
print(json.dumps({
    'runtime_number': g.G4VERSION_NUMBER,
    'runtime_string': g.G4Version,
    'classes': {name: hasattr(g, name) for name in classes},
    'methods': {owner: {name: hasattr(getattr(g, owner), name)
                        for name in names} for owner, names in methods.items()},
    'module_path': g.__file__,
}))
"""

PARTICLE_PROBE = r"""
import json
import geant4_pybind as g
p = g.G4PionPlus.PionPlusDefinition()
m = g.G4PionMinus.PionMinusDefinition()
si = g.G4NistManager.Instance().FindOrBuildMaterial('G4_Si')
vac = g.G4NistManager.Instance().FindOrBuildMaterial('G4_Galactic')
print(json.dumps({
    'pi_plus_pdg': p.GetPDGEncoding(),
    'pi_plus_mass_GeV': p.GetPDGMass()/g.GeV,
    'pi_minus_pdg': m.GetPDGEncoding(),
    'pi_minus_mass_GeV': m.GetPDGMass()/g.GeV,
    'silicon_available': si is not None,
    'near_vacuum_available': vac is not None,
}))
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_probe(code: str, env: dict[str, str]) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", code],
        input="n\n",
        text=True,
        capture_output=True,
        timeout=120,
        env=env,
        check=False,
    )
    payload = {
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.returncode == 0:
        payload["parsed"] = json.loads(result.stdout.splitlines()[-1])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geant4-data-dir", type=Path)
    args = parser.parse_args()
    env = os.environ.copy()
    if args.geant4_data_dir is not None:
        env["GEANT4_DATA_DIR"] = str(args.geant4_data_dir.resolve())
    report = {
        "created_unix": time(),
        "stage": "W0_ENVIRONMENT_PROBE_NOT_FULL_ACCEPTANCE",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "geant4_data_dir": env.get("GEANT4_DATA_DIR"),
    }
    report["import_api_probe"] = run_probe(IMPORT_PROBE, env)
    if report["import_api_probe"]["return_code"] == 0:
        module_path = Path(report["import_api_probe"]["parsed"]["module_path"])
        report["binding_binary_sha256"] = sha256(module_path)
        report["particle_material_probe"] = run_probe(PARTICLE_PROBE, env)
    report["api_and_particle_probe_passed"] = (
        report["import_api_probe"]["return_code"] == 0
        and all(report["import_api_probe"]["parsed"]["classes"].values())
        and all(
            all(value.values())
            for value in report["import_api_probe"]["parsed"]["methods"].values()
        )
        and report.get("particle_material_probe", {}).get("return_code") == 0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": report["api_and_particle_probe_passed"]}))
    return 0 if report["api_and_particle_probe_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
