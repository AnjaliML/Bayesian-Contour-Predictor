import math
import subprocess
import sys
import unittest
from pathlib import Path

import classify_drops_sized_based as sized


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "classify_drops_sized_based.py"


class SizeBasedClassifierTests(unittest.TestCase):
    def test_size_is_zero_at_and_above_critical_y(self):
        x = 10.0
        y_c = sized.critical_y(x)

        self.assertEqual(sized.drop_size(x, y_c), 0.0)
        self.assertEqual(sized.drop_size(x, y_c * 1.5), 0.0)

    def test_size_threshold_y_matches_binary_boundary(self):
        x = 4.0
        boundary = sized.size_threshold_y(x)

        self.assertEqual(sized.classify_drop(x, boundary), 0)
        self.assertEqual(sized.classify_drop(x, boundary * 0.99), 1)
        self.assertEqual(sized.classify_drop(x, boundary * 1.01), 0)

    def test_default_threshold_is_scaled_critical_y(self):
        x = 7.0
        expected = sized.critical_y(x) * (1.0 - sized.DEFAULT_SIZE_TOLERANCE / sized.SIZE_SCALE) ** 2

        self.assertAlmostEqual(sized.size_threshold_y(x), expected)

    def test_cli_prints_id_from_generic_xy(self):
        x = 6.0
        boundary = sized.size_threshold_y(x)

        drop = subprocess.run(
            [sys.executable, str(SCRIPT), str(x), str(boundary * 0.99)],
            check=True,
            text=True,
            capture_output=True,
        )
        no_drop = subprocess.run(
            [sys.executable, str(SCRIPT), str(x), str(boundary * 1.01)],
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertEqual(drop.stdout.strip(), "1")
        self.assertEqual(no_drop.stdout.strip(), "0")

    def test_cli_can_print_continuous_size(self):
        x = 8.0
        y = sized.critical_y(x) * 0.25

        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(x), str(y), "--print-size"],
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertAlmostEqual(float(result.stdout.strip()), sized.SIZE_SCALE * 0.5)


if __name__ == "__main__":
    unittest.main()
