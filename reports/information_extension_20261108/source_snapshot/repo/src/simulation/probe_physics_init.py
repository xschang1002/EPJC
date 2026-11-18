"""Isolated Windows binding probe for assigning a Geant4 physics list.

Run each method in a separate Python process because a native access violation
can leave global Geant4 state unusable.
"""

from __future__ import annotations

import argparse
import json
import os


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("method", choices=("direct", "factory", "qbbc"))
parser.add_argument("--geant4-data-dir", required=True)
parser.add_argument("--cut-timing", choices=("none", "before", "after", "managed"), default="none")
args = parser.parse_args()
os.environ["GEANT4_DATA_DIR"] = args.geant4_data_dir

import geant4_pybind as g

manager = g.G4RunManagerFactory.CreateRunManager(g.G4RunManagerType.Serial)
if args.method == "factory":
    physics = g.G4PhysListFactory().GetReferencePhysList("FTFP_BERT")
elif args.method == "qbbc":
    physics = g.QBBC()
else:
    physics = g.FTFP_BERT()
physics.SetVerboseLevel(0)
if args.cut_timing == "before":
    physics.SetDefaultCutValue(0.01 * g.mm)
print("before_SetUserInitialization", args.method, flush=True)
manager.SetUserInitialization(physics)
if args.cut_timing == "after":
    physics.SetDefaultCutValue(0.01 * g.mm)
if args.cut_timing == "managed":
    managed = manager.GetUserPhysicsList()
    print("before_cut_mm", managed.GetDefaultCutValue() / g.mm, flush=True)
    managed.SetDefaultCutValue(0.01 * g.mm)
    print("after_cut_mm", managed.GetDefaultCutValue() / g.mm, flush=True)
print(json.dumps({"method": args.method, "assigned": True, "runtime": int(g.G4VERSION_NUMBER)}), flush=True)
