import math
import unittest

import numpy as np
from scipy.integrate import quad

from models.densities import (
    TrainStandardizer,
    gain_decomposition,
    log_mixture_density,
    mixture_cdf,
    truncated_normal_cdf,
    truncated_normal_logpdf,
    truncated_normal_quantile,
    truncated_student_cdf,
    truncated_student_logpdf,
)


class DensityTests(unittest.TestCase):
    def test_truncated_normal_integrates_and_inverts(self):
        mu, sigma, upper = 0.1, 0.3, 0.7
        integral = quad(lambda y: math.exp(float(truncated_normal_logpdf(y, mu, sigma, upper))), -5, upper)[0]
        self.assertAlmostEqual(integral, 1.0, places=10)
        qs = np.array([0.01, 0.5, 0.99])
        ys = truncated_normal_quantile(qs, mu, sigma, upper)
        np.testing.assert_allclose(truncated_normal_cdf(ys, mu, sigma, upper), qs, atol=1e-10)
        self.assertTrue(np.isneginf(truncated_normal_logpdf(upper, mu, sigma, upper)))

    def test_student_integrates_and_inverts(self):
        mu, sigma, upper, df = -0.2, 0.5, 0.8, 3.0
        integral = quad(lambda y: math.exp(float(truncated_student_logpdf(y, mu, sigma, upper, df))), -1000, upper)[0]
        self.assertAlmostEqual(integral, 1.0, places=6)
        ys = np.array([-0.5, 0.1])
        self.assertTrue(np.all(np.diff(truncated_student_cdf(ys, mu, sigma, upper, df)) > 0))

    def test_mixture_weights_and_standardizer(self):
        log_weights = np.log(np.array([0.25, 0.75]))[:, None]
        component = np.log(np.array([[0.4, 0.2], [0.6, 0.8]]))
        np.testing.assert_allclose(np.exp(log_mixture_density(log_weights, component)), [0.55, 0.65])
        np.testing.assert_allclose(mixture_cdf([0.25, 0.75], [[0.2, 0.8], [0.6, 0.9]]), [0.5, 0.875])
        standardizer = TrainStandardizer(2.0, 4.0)
        np.testing.assert_allclose(standardizer.inverse(standardizer.forward([0.0, 2.0, 6.0])), [0.0, 2.0, 6.0])
        self.assertAlmostEqual(float(standardizer.original_logpdf(3.0)), 3.0 - math.log(4.0))

    def test_attribution_identities(self):
        log_weight = np.log([0.7, 0.3])
        log_anchor = np.log([0.8, 0.4])
        log_responsibility = np.log([0.5, 0.2])
        log_full = log_anchor + log_weight - log_responsibility
        result = gain_decomposition(log_full, log_anchor, np.log([0.7, 0.35]), np.log([0.6, 0.3]), log_weight, log_responsibility)
        np.testing.assert_allclose(result["direct"], log_weight - log_responsibility)


if __name__ == "__main__":
    unittest.main()
