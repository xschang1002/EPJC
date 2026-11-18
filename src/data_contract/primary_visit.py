"""First contiguous original-primary slab visit from copied Geant4 step values.

The caller must filter on the original primary track ID before calling
``observe``. This module never keeps a Geant4-owned object or pointer.
"""

from __future__ import annotations

from dataclasses import dataclass


def exit_face(position_mm: list[float], thickness_mm: float, half_width_mm: float = 10.0) -> str:
    x, y, z = position_mm
    tolerance = 1e-6
    if z >= thickness_mm / 2 - tolerance:
        return "forward"
    if z <= -thickness_mm / 2 + tolerance:
        return "backward"
    if abs(x) >= half_width_mm - tolerance or abs(y) >= half_width_mm - tolerance:
        return "lateral"
    return "unresolved_boundary"


@dataclass
class FirstVisit:
    thickness_mm: float
    entered: bool = False
    in_slab: bool = False
    first_exit_recorded: bool = False
    entry: dict | None = None
    exit: dict | None = None
    exit_face: str | None = None
    first_visit_steps: int = 0
    reentry_count: int = 0
    subsequent_visit_steps: int = 0

    def observe(self, copied_step: dict) -> str:
        """Consume a copied original-primary step and return its visit class."""
        if copied_step["pre_volume"] != "Slab":
            return "outside"
        if not self.entered:
            self.entered = True
            self.in_slab = True
            self.entry = copied_step["pre"]
        elif not self.in_slab:
            self.reentry_count += 1
            self.in_slab = True

        is_first_visit = not self.first_exit_recorded
        if is_first_visit:
            self.first_visit_steps += 1
        else:
            self.subsequent_visit_steps += 1

        post_volume = copied_step["post_volume"]
        crosses_boundary = copied_step["post_at_geom_boundary"] or (
            post_volume is not None and post_volume != "Slab"
        )
        if crosses_boundary:
            self.in_slab = False
            if is_first_visit:
                self.first_exit_recorded = True
                self.exit = copied_step["post"]
                self.exit_face = exit_face(self.exit["position_mm"], self.thickness_mm)
        return "first_visit" if is_first_visit else "subsequent_visit"

    def status(self, *, aborted: bool, original_track_ended: bool) -> str:
        if aborted:
            return "technical_abort"
        if not self.entered:
            return "never_entered"
        if self.first_exit_recorded:
            return "exited_first_visit"
        if original_track_ended:
            return "terminated_in_first_visit"
        return "incomplete_tracking_record"
