"""Real Geant4 single-slab W0 smoke run; not a production data generator.

This deliberately emits inspectable JSONL instead of the planned W1 Parquet
contract. A passing W0 run needs a separate replay and ledger audit.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_contract.primary_visit import FirstVisit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam-seed", type=int, default=20261010)
    parser.add_argument("--transport-seed", type=int, default=314159)
    parser.add_argument("--thickness-mm", type=float, default=0.1)
    parser.add_argument("--cut-mm", type=float)
    parser.add_argument("--step-fraction", type=float)
    parser.add_argument("--geant4-data-dir", type=Path)
    parser.add_argument("--input-events-jsonl", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--chunk-id", type=int)
    parser.add_argument("--record-secondary-steps", action="store_true")
    parser.add_argument("--slab-material", choices=("G4_Si", "G4_Galactic"), default="G4_Si")
    parser.add_argument("--remove-pion-process", choices=("pion-inelastic", "pion-hadElastic", "pi-Inelastic", "hadElastic"))
    args = parser.parse_args()
    if args.events <= 0 or args.thickness_mm <= 0 or (args.cut_mm is not None and args.cut_mm <= 0) or (
        args.step_fraction is not None and not 0 < args.step_fraction <= 1
    ):
        parser.error("events, thickness-mm, cut-mm, and step-fraction must be in their positive ranges")
    if args.output.exists() or args.output.with_name(args.output.name + ".partial").exists():
        parser.error("output or its partial directory already exists")
    if args.input_events_jsonl is not None and (not args.run_id or args.chunk_id is None):
        parser.error("input events require run-id and chunk-id")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_line(stream, value: dict) -> None:
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def assign_transition_targets(row: dict) -> None:
    """Populate the forward first-visit target, preserving other exits as metadata."""
    row["S1"] = False
    for name in ("p_out_GeV_c", "delta_p_GeV_c", "fractional_loss", "delta_theta_rad"):
        row[name] = None
    if row.get("entry") is None or row.get("exit") is None or row.get("exit_face") != "forward" or row.get("aborted", False):
        return
    px, py, pz = row["exit"]["momentum_GeV_c"]
    p_out = math.sqrt(px * px + py * py + pz * pz)
    p_in = float(row["p_in_GeV_c"])
    if not math.isfinite(p_out) or not math.isfinite(p_in) or p_in <= 0:
        raise ValueError("transition momenta must be finite and incident momentum positive")
    if p_out <= 0:
        return
    row["S1"] = True
    row["p_out_GeV_c"] = p_out
    row["delta_p_GeV_c"] = p_in - p_out
    row["fractional_loss"] = row["delta_p_GeV_c"] / p_in
    row["delta_theta_rad"] = math.atan2(math.hypot(px, py), pz)


def windows_memory_bytes() -> dict[str, int]:
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_memory_info.restype = wintypes.BOOL
    handle = get_current_process()
    ok = get_memory_info(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    return {"current_rss": counters.WorkingSetSize, "peak_rss": counters.PeakWorkingSetSize}


def main() -> int:
    args = parse_args()
    if args.geant4_data_dir is not None:
        os.environ["GEANT4_DATA_DIR"] = str(args.geant4_data_dir.resolve())

    # Import only after the optional dataset location is set. The binding may
    # otherwise prompt for dataset downloads during import.
    import geant4_pybind as g

    partial = args.output.with_name(args.output.name + ".partial")
    partial.mkdir(parents=True)
    event_path = partial / "events.jsonl"
    step_path = partial / "primary_steps.jsonl"
    secondary_path = partial / "secondary_summary.jsonl"
    secondary_step_path = partial / "secondary_steps.jsonl"
    started = time.perf_counter()
    beams: list[dict] = []
    if args.input_events_jsonl is None:
        beam_rng = random.Random(args.beam_seed)
        for event_id in range(args.events):
            p_bin = event_id % 7
            beams.append(
                {
                    "event_id": event_id,
                    "pdg": 211 if event_id % 2 == 0 else -211,
                    "p_bin": p_bin,
                    "p_in_GeV_c": 0.3 + 0.1 * (p_bin + beam_rng.random()),
                    "thickness_mm": args.thickness_mm,
                }
            )
    else:
        with args.input_events_jsonl.open("r", encoding="utf-8") as source:
            for event_id, line in enumerate(source):
                spec = json.loads(line)
                if not math.isclose(spec["thickness_mm"], args.thickness_mm, abs_tol=1e-12):
                    raise ValueError(f"event {event_id} slab thickness differs from chunk geometry")
                if spec["pdg"] not in (211, -211) or not 0.3 <= spec["p_in_GeV_c"] < 1.0:
                    raise ValueError(f"event {event_id} has invalid pion beam")
                beams.append({**spec, "event_id": event_id})
        if len(beams) != args.events:
            raise ValueError(f"input contains {len(beams)} events but --events is {args.events}")
        if len({b["event_uid"] for b in beams}) != len(beams):
            raise ValueError("input contains duplicate event_uid")

    def vector(v, unit: float) -> list[float]:
        return [float(v.x / unit), float(v.y / unit), float(v.z / unit)]

    def point(p) -> dict:
        return {
            "position_mm": vector(p.GetPosition(), g.mm),
            "momentum_GeV_c": vector(p.GetMomentum(), g.GeV),
            "kinetic_GeV": float(p.GetKineticEnergy() / g.GeV),
        }

    def volume_name(p) -> str | None:
        touchable = p.GetTouchable()
        if touchable is None:
            return None
        volume = touchable.GetVolume()
        return None if volume is None else str(volume.GetName())

    class Recorder:
        def __init__(self, event_stream, step_stream, secondary_stream, secondary_step_stream):
            self.event_stream = event_stream
            self.step_stream = step_stream
            self.secondary_stream = secondary_stream
            self.secondary_step_stream = secondary_step_stream
            self.current: dict | None = None
            self.visit: FirstVisit | None = None
            self.event_count = 0
            self.step_count = 0
            self.secondary_count = 0
            self.secondary_step_count = 0

        def begin(self, event_id: int) -> None:
            if self.current is not None:
                raise RuntimeError("previous event was not closed")
            self.current = {
                **beams[event_id],
                "event_uid": beams[event_id].get("event_uid", f"w0-{args.transport_seed}-{event_id}"),
                "primary_track_id": None,
                "entered": False,
                "entry": None,
                "exit": None,
                "exit_face": None,
                "primary_final_status": None,
                "slab_step_count": 0,
                "secondary_count_in_slab": 0,
            }
            self.visit = FirstVisit(args.thickness_mm)

        def primary_started(self, track) -> None:
            if self.current is None:
                raise RuntimeError("tracking action without event")
            if track.GetParentID() == 0 and abs(track.GetDefinition().GetPDGEncoding()) == 211:
                if self.current["primary_track_id"] is not None:
                    raise RuntimeError("more than one original pion primary")
                self.current["primary_track_id"] = int(track.GetTrackID())

        def primary_ended(self, track) -> None:
            if self.current is None:
                raise RuntimeError("tracking action without event")
            if track.GetTrackID() == self.current["primary_track_id"]:
                self.current["primary_final_status"] = str(track.GetTrackStatus())

        def step(self, step) -> None:
            if self.current is None:
                raise RuntimeError("stepping action without event")
            track = step.GetTrack()
            if track.GetTrackID() != self.current["primary_track_id"]:
                if args.record_secondary_steps:
                    pre = step.GetPreStepPoint()
                    post = step.GetPostStepPoint()
                    process = post.GetProcessDefinedStep()
                    creator = track.GetCreatorProcess()
                    write_json_line(self.secondary_step_stream, {
                        "event_uid": self.current["event_uid"],
                        "track_id": int(track.GetTrackID()),
                        "parent_id": int(track.GetParentID()),
                        "pdg": int(track.GetDefinition().GetPDGEncoding()),
                        "creator_process": None if creator is None else str(creator.GetProcessName()),
                        "step_index": int(track.GetCurrentStepNumber()),
                        "pre_volume": volume_name(pre),
                        "post_volume": volume_name(post),
                        "pre": point(pre),
                        "post": point(post),
                        "step_length_mm": float(step.GetStepLength() / g.mm),
                        "deposit_GeV": float(step.GetTotalEnergyDeposit() / g.GeV),
                        "process_name": None if process is None else str(process.GetProcessName()),
                        "process_type": None if process is None else str(process.GetProcessType()),
                    })
                    self.secondary_step_count += 1
                return
            pre = step.GetPreStepPoint()
            post = step.GetPostStepPoint()
            pre_name = volume_name(pre)
            post_name = volume_name(post)
            process = post.GetProcessDefinedStep()
            row = {
                "event_uid": self.current["event_uid"],
                "track_id": int(track.GetTrackID()),
                "step_index": int(track.GetCurrentStepNumber()),
                "pre_volume": pre_name,
                "post_volume": post_name,
                "post_step_status": str(post.GetStepStatus()),
                "post_at_geom_boundary": bool(post.GetStepStatus() == g.fGeomBoundary),
                "pre": point(pre),
                "post": point(post),
                "step_length_mm": float(step.GetStepLength() / g.mm),
                "deposit_GeV": float(step.GetTotalEnergyDeposit() / g.GeV),
                "process_name": None if process is None else str(process.GetProcessName()),
                "process_type": None if process is None else str(process.GetProcessType()),
            }
            visit_class = self.visit.observe(row)
            row["visit_class"] = visit_class
            write_json_line(self.step_stream, row)
            self.step_count += 1

            if visit_class == "outside":
                return
            if visit_class == "subsequent_visit":
                return
            self.current["slab_step_count"] = self.visit.first_visit_steps

            secondaries = step.GetSecondaryInCurrentStep()
            summary = []
            for secondary in secondaries:
                creator = secondary.GetCreatorProcess()
                summary.append(
                    {
                        "pdg": int(secondary.GetDefinition().GetPDGEncoding()),
                        "kinetic_GeV": float(secondary.GetKineticEnergy() / g.GeV),
                        "creator": None if creator is None else str(creator.GetProcessName()),
                    }
                )
            if summary:
                write_json_line(
                    self.secondary_stream,
                    {
                        "event_uid": self.current["event_uid"],
                        "step_index": row["step_index"],
                        "new_secondaries": summary,
                    },
                )
                self.secondary_count += len(summary)
                self.current["secondary_count_in_slab"] += len(summary)

        def end(self, event_id: int, aborted: bool) -> None:
            if self.current is None or self.current["event_id"] != event_id:
                raise RuntimeError("event ledger mismatch")
            row = self.current
            row["entered"] = self.visit.entered
            row["entry"] = self.visit.entry
            row["exit"] = self.visit.exit
            row["exit_face"] = self.visit.exit_face
            row["reentry_count"] = self.visit.reentry_count
            row["subsequent_visit_steps"] = self.visit.subsequent_visit_steps
            row["pion_mass_GeV"] = beams[event_id].get("pion_mass_GeV")
            row["aborted"] = bool(aborted)
            row["status"] = self.visit.status(aborted=bool(aborted), original_track_ended=row["primary_final_status"] is not None)
            assign_transition_targets(row)
            write_json_line(self.event_stream, row)
            self.event_count += 1
            self.current = None
            self.visit = None

    class Detector(g.G4VUserDetectorConstruction):
        def __init__(self):
            super().__init__()
            self.world = None
            self.slab = None
            self.user_limits = None

        def Construct(self):
            nist = g.G4NistManager.Instance()
            world_mat = nist.FindOrBuildMaterial("G4_Galactic")
            slab_mat = nist.FindOrBuildMaterial(args.slab_material)
            if world_mat is None or slab_mat is None:
                raise RuntimeError("required Geant4 material unavailable")
            world_logical = g.G4LogicalVolume(g.G4Box("WorldBox", 50 * g.mm, 50 * g.mm, 50 * g.mm), world_mat, "World")
            self.world = g.G4PVPlacement(None, g.G4ThreeVector(), world_logical, "World", None, False, 0, True)
            slab_logical = g.G4LogicalVolume(
                g.G4Box("SlabBox", 10 * g.mm, 10 * g.mm, args.thickness_mm * g.mm / 2),
                slab_mat,
                "Slab",
            )
            if args.step_fraction is not None:
                self.user_limits = g.G4UserLimits(args.thickness_mm * args.step_fraction * g.mm)
                slab_logical.SetUserLimits(self.user_limits)
            self.slab = g.G4PVPlacement(None, g.G4ThreeVector(), slab_logical, "Slab", world_logical, False, 0, True)
            return self.world

    class Generator(g.G4VUserPrimaryGeneratorAction):
        def __init__(self):
            super().__init__()
            self.gun = g.G4ParticleGun(1)

        def GeneratePrimaries(self, event):
            spec = beams[event.GetEventID()]
            name = "pi+" if spec["pdg"] == 211 else "pi-"
            particle = g.G4ParticleTable.GetParticleTable().FindParticle(name)
            if particle is None:
                raise RuntimeError(f"missing particle definition {name}")
            mass = float(particle.GetPDGMass())
            # GeneratePrimaries and BeginOfEventAction order is Geant4-owned.
            # Keep this copied scalar on the input record until event finalization.
            spec["pion_mass_GeV"] = mass / g.GeV
            p = spec["p_in_GeV_c"] * g.GeV
            kinetic = math.sqrt(p * p + mass * mass) - mass
            self.gun.SetParticleDefinition(particle)
            self.gun.SetParticlePosition(g.G4ThreeVector(0, 0, -10 * g.mm))
            self.gun.SetParticleMomentumDirection(g.G4ThreeVector(0, 0, 1))
            self.gun.SetParticleEnergy(kinetic)
            self.gun.GeneratePrimaryVertex(event)

    class EventAction(g.G4UserEventAction):
        def __init__(self, recorder):
            super().__init__()
            self.recorder = recorder

        def BeginOfEventAction(self, event):
            self.recorder.begin(int(event.GetEventID()))

        def EndOfEventAction(self, event):
            self.recorder.end(int(event.GetEventID()), bool(event.IsAborted()))

    class TrackingAction(g.G4UserTrackingAction):
        def __init__(self, recorder):
            super().__init__()
            self.recorder = recorder

        def PreUserTrackingAction(self, track):
            self.recorder.primary_started(track)

        def PostUserTrackingAction(self, track):
            self.recorder.primary_ended(track)

    class SteppingAction(g.G4UserSteppingAction):
        def __init__(self, recorder):
            super().__init__()
            self.recorder = recorder

        def UserSteppingAction(self, step):
            self.recorder.step(step)

    with event_path.open("w", encoding="utf-8", newline="\n") as events, step_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as steps, secondary_path.open("w", encoding="utf-8", newline="\n") as secondary, secondary_step_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as secondary_steps:
        recorder = Recorder(events, steps, secondary, secondary_steps)
        manager = g.G4RunManagerFactory.CreateRunManager(g.G4RunManagerType.Serial)
        detector = Detector()
        physics = g.FTFP_BERT()
        physics.SetVerboseLevel(0)
        if args.step_fraction is not None:
            physics.RegisterPhysics(g.G4StepLimiterPhysics())
        # W0 uses the unmodified reference-list cut. The requested 0.01 mm
        # pilot candidate must be tested through a safe binding path in W2.
        # Geant4 11.4 forbids constructing user actions before the physics
        # list has been assigned to the run manager.
        manager.SetUserInitialization(physics)
        manager.SetUserInitialization(detector)
        generator = Generator()
        event_action = EventAction(recorder)
        tracking_action = TrackingAction(recorder)
        stepping_action = SteppingAction(recorder)
        manager.SetUserAction(generator)
        manager.SetUserAction(event_action)
        manager.SetUserAction(tracking_action)
        manager.SetUserAction(stepping_action)
        g.G4Random.setTheEngine(g.MTwistEngine())
        g.G4Random.setTheSeed(args.transport_seed)
        g.G4Random.saveEngineStatus(str(partial / "transport_rng_start.conf"))
        if args.cut_mm is not None:
            g.G4UImanager.GetUIpointer().ApplyCommand(f"/run/setCut {args.cut_mm:.12g} mm")
        manager.Initialize()
        if args.cut_mm is not None:
            manager.GetUserPhysicsList().DumpCutValuesTable()
        process_table = {}
        pion_masses = {}
        removed_processes = []
        if args.remove_pion_process is not None:
            for name in ("pi+", "pi-"):
                particle = g.G4ParticleTable.GetParticleTable().FindParticle(name)
                process_manager = particle.GetProcessManager()
                for process in list(process_manager.GetProcessList()):
                    process_name = str(process.GetProcessName())
                    wanted = (
                        process_name in ("pi+Inelastic", "pi-Inelastic") if args.remove_pion_process == "pion-inelastic" else
                        process_name == "hadElastic" if args.remove_pion_process in ("pion-hadElastic", "hadElastic") else
                        process_name == args.remove_pion_process
                    )
                    if wanted:
                        process_manager.RemoveProcess(process)
                        removed_processes.append({"particle": name, "process": process_name})
            manager.PhysicsHasBeenModified()
        for name in ("pi+", "pi-"):
            particle = g.G4ParticleTable.GetParticleTable().FindParticle(name)
            if particle is None or particle.GetProcessManager() is None:
                raise RuntimeError(f"missing initialized process manager for {name}")
            pion_masses[name] = float(particle.GetPDGMass() / g.GeV)
            process_table[name] = [
                {"name": str(process.GetProcessName()), "type": str(process.GetProcessType())}
                for process in particle.GetProcessManager().GetProcessList()
            ]
        initialized_seconds = time.perf_counter() - started
        manager.BeamOn(args.events)
        if recorder.event_count != args.events:
            raise RuntimeError(f"ledger has {recorder.event_count} rows for {args.events} events")

    manifest = {
        "stage": (
            "W2_NEAR_VACUUM_CONTROL_NOT_TRAINING" if args.slab_material == "G4_Galactic" else
            "W0_REAL_GEANT4_SMOKE_NOT_PRODUCTION" if args.input_events_jsonl is None else
            "W1_REAL_GEANT4_CHUNK_CANDIDATE_NOT_PRODUCTION"
        ),
        "events": args.events,
        "ledger_rows": recorder.event_count,
        "primary_step_rows": recorder.step_count,
        "new_secondaries_in_slab": recorder.secondary_count,
        "secondary_step_rows": recorder.secondary_step_count,
        "full_secondary_step_policy": "all_secondary_tracks_all_volumes" if args.record_secondary_steps else "not_recorded",
        "python": sys.version,
        "geant4_binding": "geant4-pybind==0.1.3",
        "geant4_runtime_number": int(g.G4VERSION_NUMBER),
        "geant4_runtime_string": str(g.G4Version),
        "physics_list": "FTFP_BERT",
        "cut_configuration": "unmodified_FTFP_BERT_default_W0_only" if args.cut_mm is None else "UI_run_setCut_exploratory_not_production",
        "requested_cut_mm": args.cut_mm,
        "cut_table_dumped_to_log": args.cut_mm is not None,
        "step_fraction_candidate": args.step_fraction,
        "max_slab_step_mm_requested": None if args.step_fraction is None else args.thickness_mm * args.step_fraction,
        "world_material": "G4_Galactic",
        "slab_material": args.slab_material,
        "pion_masses_GeV": pion_masses,
        "pion_process_table": process_table,
        "removed_pion_processes": removed_processes,
        "thickness_mm": args.thickness_mm,
        "beam_seed": args.beam_seed,
        "beam_input": None if args.input_events_jsonl is None else str(args.input_events_jsonl.resolve()),
        "beam_input_sha256": None if args.input_events_jsonl is None else sha256(args.input_events_jsonl),
        "run_id": args.run_id,
        "chunk_id": args.chunk_id,
        "transport_seed": args.transport_seed,
        "geant4_data_dir": os.environ.get("GEANT4_DATA_DIR"),
        "initialization_seconds": initialized_seconds,
        "total_seconds": time.perf_counter() - started,
        "memory_bytes": windows_memory_bytes(),
        "files_sha256": {
            p.name: sha256(p)
            for p in (event_path, step_path, secondary_path, secondary_step_path, partial / "transport_rng_start.conf")
        },
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    partial.rename(args.output)
    print(json.dumps({"output": str(args.output), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
