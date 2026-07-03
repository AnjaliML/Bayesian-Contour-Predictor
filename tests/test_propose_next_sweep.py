import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import propose_next_sweep as sweep


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "propose_next_sweep.py"


class ProposeNextSweepTests(unittest.TestCase):
    def write_csv(self, directory, name, text):
        path = Path(directory) / name
        path.write_text(text, encoding="utf-8")
        return path

    def read_rows(self, path):
        with Path(path).open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_aggregates_repeats_and_ignores_stale_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                "sweep.csv",
                "\n".join(
                    [
                        "caseId,x,y,id",
                        "1,8,0.075,0",
                        "2,8,0.075,1",
                        "3,9,0.090,-1",
                        "4,10,0.120,1",
                    ]
                ),
            )

            observations, case_ids = sweep.read_csv_files(
                [path], case_col="caseId", x_col="x", y_col="y", label_col="id"
            )
            aggregates = sweep.aggregate_observations(observations)

            self.assertEqual(case_ids, ["1", "2", "3", "4"])
            self.assertEqual(len(observations), 3)
            self.assertIn(sweep.AggregatePoint(x=8.0, y=0.075, n=2, k=1), aggregates)

    def test_cli_writes_deterministic_process_agnostic_proposals(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.write_csv(
                directory,
                "sweep.csv",
                "\n".join(
                    [
                        "caseId,x,y,id",
                        "1,0.0,0.20,1",
                        "2,0.0,0.80,0",
                        "3,0.5,0.35,1",
                        "4,0.5,0.65,-1",
                        "5,1.0,0.30,1",
                        "6,1.0,0.90,0",
                    ]
                ),
            )
            out1 = Path(directory) / "next-1.csv"
            out2 = Path(directory) / "next-2.csv"
            base_command = [
                sys.executable,
                str(SCRIPT),
                str(data),
                "--n-simulations",
                "4",
                "--n-new",
                "3",
                "--n-repeats",
                "1",
                "--seed",
                "17",
                "--x-min",
                "0",
                "--x-max",
                "1",
                "--y-min",
                "0.1",
                "--y-max",
                "1.0",
                "--grid-size",
                "9",
                "--posterior-samples",
                "6",
            ]

            subprocess.run([*base_command, "--outfile", str(out1)], check=True)
            subprocess.run([*base_command, "--outfile", str(out2)], check=True)

            self.assertEqual(out1.read_text(encoding="utf-8"), out2.read_text(encoding="utf-8"))
            rows = self.read_rows(out1)
            self.assertEqual(len(rows), 4)
            self.assertEqual(set(rows[0]), set(sweep.PROPOSAL_COLUMNS))
            self.assertEqual({row["id"] for row in rows}, {"-1"})
            self.assertEqual(sum(row["proposal_type"] == "new" for row in rows), 3)
            self.assertEqual(sum(row["proposal_type"] == "repeat" for row in rows), 1)
            self.assertTrue(all(row["reason"] for row in rows))

    def test_legacy_columns_only(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.write_csv(
                directory,
                "sweep.csv",
                "\n".join(
                    [
                        "caseId,x,y,id",
                        "1,0,0,0",
                        "2,1,1,1",
                        "3,1,0,1",
                    ]
                ),
            )
            out = Path(directory) / "legacy.csv"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(data),
                    "--outfile",
                    str(out),
                    "--n-simulations",
                    "2",
                    "--legacy-columns-only",
                    "--grid-size",
                    "7",
                    "--posterior-samples",
                    "3",
                ],
                check=True,
            )

            rows = self.read_rows(out)
            self.assertEqual(list(rows[0]), sweep.LEGACY_COLUMNS)
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["id"] for row in rows}, {"-1"})

    def test_physical_columns_can_be_mapped_to_generic_xy(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.write_csv(
                directory,
                "drop_sweep.csv",
                "\n".join(
                    [
                        "caseId,Rr,Oh,id",
                        "1,4,0.020,1",
                        "2,4,0.090,0",
                        "3,8,0.025,1",
                        "4,8,0.120,0",
                    ]
                ),
            )
            out = Path(directory) / "next.csv"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(data),
                    "--outfile",
                    str(out),
                    "--x-col",
                    "Rr",
                    "--y-col",
                    "Oh",
                    "--mode",
                    "monotone-y",
                    "--y-scale",
                    "log10",
                    "--y-min",
                    "0.01",
                    "--y-max",
                    "0.2",
                    "--n-simulations",
                    "3",
                    "--grid-size",
                    "7",
                    "--posterior-samples",
                    "3",
                    "--seed",
                    "4",
                ],
                check=True,
            )

            rows = self.read_rows(out)
            self.assertEqual(len(rows), 3)
            self.assertIn("x", rows[0])
            self.assertIn("y", rows[0])
            self.assertNotIn("Rr", rows[0])
            self.assertNotIn("Oh", rows[0])

    def test_monotone_decreasing_probability_falls_as_y_increases(self):
        low_y = sweep.monotone_probability(
            0.02,
            0.05,
            direction="decreasing",
            y_scale="log10",
            transition_width=0.10,
            label_noise=0.02,
        )
        high_y = sweep.monotone_probability(
            0.10,
            0.05,
            direction="decreasing",
            y_scale="log10",
            transition_width=0.10,
            label_noise=0.02,
        )

        self.assertGreater(low_y, high_y)

    def test_monotone_contour_refines_between_grid_points(self):
        domain = sweep.Domain(x_min=0.0, x_max=1.0, y_min=0.001, y_max=0.1)
        aggregates = [
            sweep.AggregatePoint(x=0.5, y=0.006, n=1, k=1),
            sweep.AggregatePoint(x=0.5, y=0.008, n=1, k=0),
        ]
        config = sweep.ModelConfig(
            mode="monotone-y",
            monotone_direction="decreasing",
            x_scale="linear",
            y_scale="log10",
            transition_width=0.05,
            label_noise=0.02,
            length_scale_x=1.0,
            length_scale_y=0.4,
            prior_alpha=1.0,
            prior_beta=1.0,
            grid_size=5,
            posterior_samples=0,
        )

        estimate = sweep.estimate_monotone_y_c(0.5, aggregates, domain, config)

        self.assertGreater(estimate, 0.006)
        self.assertLess(estimate, 0.008)
        self.assertNotAlmostEqual(estimate, 0.01)

    def test_log_x_scale_uses_geometric_candidate_spacing(self):
        domain = sweep.Domain(x_min=1.0, x_max=100.0, y_min=0.001, y_max=0.1)
        aggregates = [
            sweep.AggregatePoint(x=1.0, y=0.01, n=1, k=1),
            sweep.AggregatePoint(x=100.0, y=0.04, n=1, k=0),
        ]
        config = sweep.ModelConfig(
            mode="monotone-y",
            monotone_direction="decreasing",
            x_scale="log10",
            y_scale="log10",
            transition_width=0.10,
            label_noise=0.02,
            length_scale_x=0.4,
            length_scale_y=0.4,
            prior_alpha=1.0,
            prior_beta=1.0,
            grid_size=5,
            posterior_samples=0,
        )

        candidates = sweep.candidate_x_values(domain, aggregates, config, 3)

        self.assertTrue(any(abs(value - 10.0) < 1e-9 for value in candidates))


if __name__ == "__main__":
    unittest.main()
