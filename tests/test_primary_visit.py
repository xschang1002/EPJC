"""Synthetic step fixtures test software state transitions, not pion physics."""

import unittest

from data_contract.primary_visit import FirstVisit


def point(z: float, x: float = 0.0) -> dict:
    return {"position_mm": [x, 0.0, z], "momentum_GeV_c": [0.0, 0.0, 0.5]}


def step(pre_volume: str, post_volume: str | None, z0: float, z1: float, *, boundary=False, x1=0.0) -> dict:
    return {
        "pre_volume": pre_volume,
        "post_volume": post_volume,
        "pre": point(z0),
        "post": point(z1, x1),
        "post_at_geom_boundary": boundary,
    }


class FirstVisitTests(unittest.TestCase):
    def test_forward_first_exit_then_reentry_is_auxiliary(self):
        visit = FirstVisit(0.1)
        self.assertEqual(visit.observe(step("World", "Slab", -1.0, -0.05, boundary=True)), "outside")
        self.assertEqual(visit.observe(step("Slab", "Slab", -0.05, 0.0)), "first_visit")
        self.assertEqual(visit.observe(step("Slab", "World", 0.0, 0.05, boundary=True)), "first_visit")
        first_exit = visit.exit
        self.assertEqual(visit.observe(step("Slab", "World", 0.05, -0.05, boundary=True)), "subsequent_visit")
        self.assertEqual(visit.reentry_count, 1)
        self.assertIs(visit.exit, first_exit)
        self.assertEqual(visit.exit_face, "forward")
        self.assertEqual(visit.status(aborted=False, original_track_ended=True), "exited_first_visit")

    def test_backward_and_lateral_exit(self):
        backward = FirstVisit(0.3)
        backward.observe(step("Slab", "World", 0.0, -0.15, boundary=True))
        self.assertEqual(backward.exit_face, "backward")
        lateral = FirstVisit(1.0)
        lateral.observe(step("Slab", "World", 0.0, 0.0, boundary=True, x1=10.0))
        self.assertEqual(lateral.exit_face, "lateral")

    def test_absorption_and_technical_abort_are_distinct(self):
        visit = FirstVisit(0.1)
        visit.observe(step("Slab", "Slab", -0.05, -0.02))
        self.assertEqual(visit.status(aborted=False, original_track_ended=True), "terminated_in_first_visit")
        self.assertEqual(visit.status(aborted=True, original_track_ended=True), "technical_abort")
        self.assertIsNone(visit.exit)


if __name__ == "__main__":
    unittest.main()
