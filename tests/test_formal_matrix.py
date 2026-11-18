"""Numerical contract tests for the formal matrix (synthetic unit fixtures only)."""

import unittest

import numpy as np

from models.fit_formal_matrix import predict_model


class FormalMatrixTests(unittest.TestCase):
    def test_all_models_return_finite_density_inside_support(self):
        z = np.asarray([0.1])
        upper = np.asarray([2.0])
        anchor = np.asarray([0.0])
        x = np.asarray([[1.0, 0.2, 1.0, -0.5]])
        specs = {
            "anchor_independent": {"params": [0.0]},
            "shared_gaussian": {"params": [0.0, 0.0]},
            "conditional_gaussian": {"params": [0.0, 0.0, 0.0, 0.0, 0.0]},
            "conditional_student_t_df3": {"params": [0.0, 0.0, 0.0, 0.0, 0.0]},
            "frozen_anchor_k2": {"params": [0.0, 0.0, 0.0, 0.0, 0.0]},
            "joint_anchor_residual_k2": {"params": [0.0, 0.1, 0.0, 0.0, 0.0]},
            "conditional_gate_anchor_k2": {"params": [0.0] * 12 + [0.0, 0.0]},
            "learned_center_k2": {"params": [0.0] * 11},
            "unanchored_mdn_k2": {"params": [0.0] * 14},
        }
        for name, spec in specs.items():
            value = predict_model(name, spec, z, upper, anchor, x)
            self.assertTrue(np.isfinite(value).all(), name)


if __name__ == "__main__":
    unittest.main()
