import collections
import unittest

from data_contract.beam_schedule import CELLS, MOMENTUM_BINS, chunk_transport_seeds, event_spec


class BeamScheduleTests(unittest.TestCase):
    def test_42_cells_once_per_complete_block(self):
        events = [event_spec("test", i, 1234) for i in range(42)]
        cells = {(row.momentum_bin, row.pdg, row.thickness_mm) for row in events}
        self.assertEqual(cells, set(CELLS))
        self.assertEqual(len(cells), 42)

    def test_chunk_boundaries_do_not_change_events(self):
        full = [event_spec("run-a", i, 1234) for i in range(1000)]
        chunks = [
            [event_spec("run-a", i, 1234) for i in range(start, min(start + 73, 1000))]
            for start in range(0, 1000, 73)
        ]
        self.assertEqual(full, [row for chunk in chunks for row in chunk])
        counts = collections.Counter((e.momentum_bin, e.pdg, e.thickness_mm) for e in full)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        for event in full:
            low, high = MOMENTUM_BINS[event.momentum_bin]
            self.assertGreaterEqual(event.p_in_GeV_c, low)
            self.assertLess(event.p_in_GeV_c, high)

    def test_distinct_worker_seeds(self):
        seeds = chunk_transport_seeds(20261010, list(range(1000)))
        self.assertEqual(len(set(seeds.values())), 1000)
        with self.assertRaises(ValueError):
            chunk_transport_seeds(20261010, [1, 1])


if __name__ == "__main__":
    unittest.main()
