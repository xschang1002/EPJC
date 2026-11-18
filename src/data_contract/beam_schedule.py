"""Deterministic 42-cell beam design, independent of Geant4 transport.

This is an input schedule, not synthetic physics data. Chunk boundaries cannot
change event membership or the sampled incident momentum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import random


PDG_CODES = (211, -211)
THICKNESSES_MM = (0.1, 0.3, 1.0)
MOMENTUM_BINS = tuple((0.3 + 0.1 * i, 0.4 + 0.1 * i) for i in range(7))
CELLS = tuple(itertools.product(range(7), PDG_CODES, THICKNESSES_MM))
assert len(CELLS) == 42


def derive_seed(master_seed: int, *labels: object) -> int:
    digest = hashlib.sha256(
        "|".join([str(master_seed), *(str(label) for label in labels)]).encode("utf-8")
    ).digest()
    # CLHEP/Windows seed entry points may use 32-bit long. Reserve zero.
    return int.from_bytes(digest[:8], "big") % (2**31 - 2) + 1


@dataclass(frozen=True)
class BeamEvent:
    event_uid: str
    event_index: int
    momentum_bin: int
    pdg: int
    thickness_mm: float
    p_in_GeV_c: float

    def as_record(self) -> dict:
        return asdict(self)


def event_spec(run_id: str, event_index: int, beam_seed: int) -> BeamEvent:
    if event_index < 0:
        raise ValueError("event_index must be nonnegative")
    block, offset = divmod(event_index, len(CELLS))
    permutation = list(CELLS)
    random.Random(derive_seed(beam_seed, "cell-block", block)).shuffle(permutation)
    bin_index, pdg, thickness = permutation[offset]
    p_low, p_high = MOMENTUM_BINS[bin_index]
    draw = random.Random(derive_seed(beam_seed, "momentum", event_index)).random()
    p = p_low + (p_high - p_low) * draw
    return BeamEvent(
        event_uid=f"{run_id}:{event_index}",
        event_index=event_index,
        momentum_bin=bin_index,
        pdg=pdg,
        thickness_mm=thickness,
        p_in_GeV_c=p,
    )


def chunk_transport_seeds(master_seed: int, chunk_ids: list[int]) -> dict[int, int]:
    seeds = {chunk_id: derive_seed(master_seed, "transport-chunk", chunk_id) for chunk_id in chunk_ids}
    if len(seeds) != len(chunk_ids):
        raise ValueError("duplicate chunk ID")
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("transport seed collision; choose a new master seed or chunk catalog")
    return seeds

