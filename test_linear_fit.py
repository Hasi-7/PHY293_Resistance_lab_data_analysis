"""Numerical checks for the uncertainty-aware linear fitting workflow."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from linear_fit import fit_line, read_measurements


class LinearFitTests(unittest.TestCase):
    def test_y_dominated_fit_matches_analytic_equal_weight_least_squares(self):
        x = np.array([0., 1., 2., 3., 4.])
        y = np.array([1.1, 2.8, 5.2, 7.1, 9.0])
        sx = np.full(5, 1e-9)
        sy = np.full(5, 0.2)
        beta, errors, chi2, reduced, residuals = fit_line(x, y, sx, sy)

        # In the negligible-x-error limit, the PDFs' unweighted equations
        # apply: var(m)=sigma_y^2/Sxx and
        # var(b)=sigma_y^2*(1/N + x_mean^2/Sxx).
        slope, intercept = np.polyfit(x, y, 1)
        sxx = np.sum((x - np.mean(x)) ** 2)
        expected_errors = (0.2 / np.sqrt(sxx),
                           0.2 * np.sqrt(1 / len(x) + np.mean(x) ** 2 / sxx))
        expected_residuals = y - (slope * x + intercept)
        np.testing.assert_allclose(beta, [slope, intercept], atol=1e-6)
        np.testing.assert_allclose(errors, expected_errors, rtol=1e-5)
        np.testing.assert_allclose(residuals, expected_residuals, atol=1e-6)
        self.assertAlmostEqual(chi2, np.sum((expected_residuals / sy) ** 2), places=5)
        self.assertAlmostEqual(reduced, chi2 / 3)

    def test_x_uncertainty_changes_fit_and_chi_squared(self):
        x = np.array([0., 1., 2., 3., 4.])
        y = np.array([0., 2., 4., 6., 10.])
        sy = np.full(5, 0.1)
        ordinary = np.polyfit(x, y, 1)[0]
        beta, errors, chi2, reduced, _ = fit_line(x, y, np.full(5, 0.5), sy)
        self.assertGreater(abs(beta[0] - ordinary), 0.05)
        self.assertGreater(errors[0], 0)
        self.assertGreater(errors[1], 0)
        # For a linear model, projecting a point onto the best-fit line gives
        # (y-mx-b)^2/(sigma_y^2 + m^2*sigma_x^2) for its chi^2 contribution.
        expected_chi2 = np.sum((y - (beta[0] * x + beta[1])) ** 2 /
                               (sy ** 2 + (beta[0] * 0.5) ** 2))
        self.assertAlmostEqual(chi2, expected_chi2, places=5)
        self.assertAlmostEqual(reduced, chi2 / 3)

    def test_invalid_csv_uncertainty_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("x,y,sigma_x,sigma_y\n0,1,0.1,0.2\n1,2,0,0.2\n2,3,0.1,0.2\n")
            with self.assertRaisesRegex(ValueError, "row 3: uncertainties must be positive"):
                read_measurements(path)

    def test_cli_creates_fit_and_residual_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "readings.csv"
            csv_path.write_text("x,y,sigma_x,sigma_y\n0,1.1,0.1,0.2\n"
                                "1,3.0,0.1,0.2\n2,5.2,0.1,0.2\n3,7.0,0.1,0.2\n")
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("linear_fit.py")),
                 str(csv_path), "--x-unit", "A", "--y-unit", "V"],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("reduced chi^2", completed.stdout)
            self.assertIn("V/A", completed.stdout)
            self.assertGreater((Path(directory) / "readings_fit.png").stat().st_size, 1000)
            self.assertEqual((Path(directory) / "readings_fit.txt").read_text(encoding="utf-8"),
                             completed.stdout)


if __name__ == "__main__":
    unittest.main()
