import csv
import math
import random
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
            contour_fit="local-linear",
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

    def test_boundary_focus_adds_wider_y_bracket_probes(self):
        domain = sweep.Domain(x_min=1.0, x_max=100.0, y_min=0.001, y_max=0.1)
        config = sweep.ModelConfig(
            mode="monotone-y",
            monotone_direction="decreasing",
            contour_fit="local-linear",
            x_scale="log10",
            y_scale="log10",
            transition_width=0.10,
            label_noise=0.02,
            length_scale_x=0.18,
            length_scale_y=0.4,
            prior_alpha=1.0,
            prior_beta=1.0,
            grid_size=5,
            posterior_samples=0,
        )
        contour = sweep.ContourEstimate(
            y_c_pred=0.006,
            y_c_q05=0.006,
            y_c_q95=0.006,
            y_c_std=0.0,
        )

        interior = sweep.y_probe_values_near_contour(
            contour, domain, config, edge_focus=0.0
        )
        boundary = sweep.y_probe_values_near_contour(
            contour, domain, config, edge_focus=1.0
        )

        self.assertGreater(len(boundary), len(interior))
        self.assertLess(min(value for value, _ in boundary), min(value for value, _ in interior))

    def test_inverse_locator_clamps_y_outside_contour_to_nearest_x_edge(self):
        domain = sweep.Domain(x_min=1.0, x_max=100.0, y_min=0.001, y_max=0.1)
        config = sweep.ModelConfig(
            mode="monotone-y",
            monotone_direction="decreasing",
            contour_fit="local-linear",
            x_scale="log10",
            y_scale="log10",
            transition_width=0.10,
            label_noise=0.02,
            length_scale_x=0.18,
            length_scale_y=0.4,
            prior_alpha=1.0,
            prior_beta=1.0,
            grid_size=5,
            posterior_samples=0,
        )
        contour_path = [
            (1.0, sweep.ContourEstimate(0.006, 0.006, 0.006, 0.0)),
            (10.0, sweep.ContourEstimate(0.03, 0.03, 0.03, 0.0)),
            (100.0, sweep.ContourEstimate(0.032, 0.032, 0.032, 0.0)),
        ]

        candidates = sweep.x_candidates_for_y_level(0.004, contour_path, domain, config)

        self.assertEqual(candidates, [1.0])

    def test_boundary_contour_is_not_pulled_inward_by_one_sided_kernel(self):
        domain = sweep.Domain(x_min=1.0, x_max=100.0, y_min=0.001, y_max=0.1)

        def threshold(x):
            return 0.0045 + 0.02 * (1.0 - math.exp(-0.35 * (x - 1.0)))

        observations = []
        case_id = 1
        for index in range(50):
            fraction = index / 49
            y = 10 ** (
                math.log10(0.0015)
                + (math.log10(0.012) - math.log10(0.0015)) * fraction
            )
            observations.append(
                sweep.Observation(
                    str(case_id),
                    1.0,
                    y,
                    1 if y < threshold(1.0) else 0,
                )
            )
            case_id += 1
        for index in range(30):
            fraction = index / 29
            x = 10 ** (
                math.log10(1.05)
                + (math.log10(15.0) - math.log10(1.05)) * fraction
            )
            y_c = threshold(x)
            for multiplier in [0.8, 0.95, 1.05, 1.25]:
                y = y_c * multiplier
                observations.append(
                    sweep.Observation(
                        str(case_id),
                        x,
                        y,
                        1 if y < threshold(x) else 0,
                    )
                )
                case_id += 1
        config = sweep.ModelConfig(
            mode="monotone-y",
            monotone_direction="decreasing",
            contour_fit="local-linear",
            x_scale="log10",
            y_scale="log10",
            transition_width=0.10,
            label_noise=0.02,
            length_scale_x=0.18,
            length_scale_y=0.4,
            prior_alpha=1.0,
            prior_beta=1.0,
            grid_size=15,
            posterior_samples=0,
        )

        estimate = sweep.contour_estimate(
            1.0,
            sweep.aggregate_observations(observations),
            domain,
            config,
            rng=random.Random(1),
        ).y_c_pred

        self.assertLess(abs(math.log10(estimate) - math.log10(threshold(1.0))), 0.02)

    def test_label_bracket_midpoints_use_transformed_y(self):
        domain = sweep.Domain(x_min=1.0, x_max=100.0, y_min=0.001, y_max=0.1)
        aggregates = [
            sweep.AggregatePoint(x=1.0, y=0.004, n=1, k=1),
            sweep.AggregatePoint(x=1.0, y=0.008, n=1, k=0),
        ]
        config = sweep.ModelConfig(
            mode="monotone-y",
            monotone_direction="decreasing",
            contour_fit="local-linear",
            x_scale="log10",
            y_scale="log10",
            transition_width=0.04,
            label_noise=0.005,
            length_scale_x=0.18,
            length_scale_y=0.4,
            prior_alpha=1.0,
            prior_beta=1.0,
            grid_size=15,
            posterior_samples=0,
        )

        candidates = sweep.bracket_midpoint_candidates(aggregates, domain, config)

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0][0], 1.0)
        self.assertAlmostEqual(candidates[0][1], math.sqrt(0.004 * 0.008))
        self.assertGreater(candidates[0][2], 0.0)

    def test_proposals_prioritize_observed_label_brackets(self):
        domain = sweep.Domain(x_min=1.0, x_max=100.0, y_min=0.001, y_max=0.1)
        observations = [
            sweep.Observation("1", 1.0, 0.004, 1),
            sweep.Observation("2", 1.0, 0.008, 0),
            sweep.Observation("3", 4.0, 0.014, 1),
            sweep.Observation("4", 4.0, 0.024, 0),
            sweep.Observation("5", 20.0, 0.024, 1),
            sweep.Observation("6", 20.0, 0.040, 0),
            sweep.Observation("7", 100.0, 0.024, 1),
            sweep.Observation("8", 100.0, 0.050, 0),
        ]
        config = sweep.ModelConfig(
            mode="monotone-y",
            monotone_direction="decreasing",
            contour_fit="local-linear",
            x_scale="log10",
            y_scale="log10",
            transition_width=0.04,
            label_noise=0.005,
            length_scale_x=0.18,
            length_scale_y=0.4,
            prior_alpha=1.0,
            prior_beta=1.0,
            grid_size=15,
            posterior_samples=0,
        )

        proposals = sweep.propose_next_batch(
            observations,
            domain=domain,
            config=config,
            n_simulations=4,
            n_new=4,
            n_repeats=0,
            seed=3,
        )

        self.assertTrue(
            any("bisects observed label bracket" in proposal.reason for proposal in proposals)
        )

    def test_one_sided_edge_label_expands_edge_bracket(self):
        domain = sweep.Domain(x_min=1.0, x_max=100.0, y_min=0.001, y_max=0.1)
        aggregates = [
            sweep.AggregatePoint(x=1.0, y=0.006, n=1, k=0),
            sweep.AggregatePoint(x=1.3, y=0.006, n=1, k=1),
        ]
        config = sweep.ModelConfig(
            mode="monotone-y",
            monotone_direction="decreasing",
            contour_fit="adaptive-linear",
            x_scale="log10",
            y_scale="log10",
            transition_width=0.04,
            label_noise=0.005,
            length_scale_x=0.18,
            length_scale_y=0.4,
            prior_alpha=1.0,
            prior_beta=1.0,
            grid_size=15,
            posterior_samples=0,
        )

        candidates = sweep.edge_bracket_candidates(aggregates, domain, config)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][0], 1.0)
        self.assertLess(candidates[0][1], 0.006)

    def test_tight_exact_bracket_constrains_contour_estimate(self):
        domain = sweep.Domain(x_min=1.0, x_max=100.0, y_min=0.001, y_max=0.1)
        aggregates = [
            sweep.AggregatePoint(x=1.0, y=0.00449, n=1, k=1),
            sweep.AggregatePoint(x=1.0, y=0.00451, n=1, k=0),
            sweep.AggregatePoint(x=2.0, y=0.010, n=1, k=1),
            sweep.AggregatePoint(x=2.0, y=0.014, n=1, k=0),
        ]
        config = sweep.ModelConfig(
            mode="monotone-y",
            monotone_direction="decreasing",
            contour_fit="adaptive-linear",
            x_scale="log10",
            y_scale="log10",
            transition_width=0.04,
            label_noise=0.005,
            length_scale_x=0.18,
            length_scale_y=0.4,
            prior_alpha=1.0,
            prior_beta=1.0,
            grid_size=15,
            posterior_samples=0,
        )

        estimate = sweep.estimate_monotone_y_c(1.0, aggregates, domain, config)

        self.assertAlmostEqual(estimate, math.sqrt(0.00449 * 0.00451))

    def test_stratified_selection_keeps_multiple_x_regions(self):
        domain = sweep.Domain(x_min=1.0, x_max=100.0, y_min=0.001, y_max=0.1)
        config = sweep.ModelConfig(
            mode="monotone-y",
            monotone_direction="decreasing",
            contour_fit="local-linear",
            x_scale="log10",
            y_scale="log10",
            transition_width=0.04,
            label_noise=0.005,
            length_scale_x=0.18,
            length_scale_y=0.4,
            prior_alpha=1.0,
            prior_beta=1.0,
            grid_size=15,
            posterior_samples=0,
        )
        contour = sweep.ContourEstimate(0.01, 0.01, 0.01, 0.0)
        clustered = [
            sweep.Proposal(
                x=1.0 + index * 0.001,
                y=0.004 + index * 0.0001,
                proposal_type="new",
                score=10.0 - index * 0.01,
                p_positive_pred=0.5,
                contour=contour,
                n_existing=0,
                k_existing=0,
                reason="clustered",
            )
            for index in range(10)
        ]
        spread = [
            sweep.Proposal(
                x=x,
                y=0.02,
                proposal_type="new",
                score=1.0,
                p_positive_pred=0.5,
                contour=contour,
                n_existing=0,
                k_existing=0,
                reason="spread",
            )
            for x in (4.0, 20.0, 80.0)
        ]

        selected = sweep.select_stratified_proposals(clustered + spread, domain, config, 4)
        bins = {
            sweep.x_bin_index(proposal.x, domain, config, 4)
            for proposal in selected
        }

        self.assertGreater(len(bins), 1)

    def test_log_x_scale_uses_geometric_candidate_spacing(self):
        domain = sweep.Domain(x_min=1.0, x_max=100.0, y_min=0.001, y_max=0.1)
        aggregates = [
            sweep.AggregatePoint(x=1.0, y=0.01, n=1, k=1),
            sweep.AggregatePoint(x=100.0, y=0.04, n=1, k=0),
        ]
        config = sweep.ModelConfig(
            mode="monotone-y",
            monotone_direction="decreasing",
            contour_fit="local-linear",
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

    def test_candidate_x_midpoints_are_bounded_for_long_campaigns(self):
        domain = sweep.Domain(x_min=1.0, x_max=100.0, y_min=0.001, y_max=0.1)
        aggregates = [
            sweep.AggregatePoint(x=1.0 + index, y=0.01, n=1, k=index % 2)
            for index in range(80)
        ]
        config = sweep.ModelConfig(
            mode="monotone-y",
            monotone_direction="decreasing",
            contour_fit="local-linear",
            x_scale="linear",
            y_scale="log10",
            transition_width=0.04,
            label_noise=0.005,
            length_scale_x=10.0,
            length_scale_y=0.4,
            prior_alpha=1.0,
            prior_beta=1.0,
            grid_size=5,
            posterior_samples=0,
        )

        candidates = sweep.candidate_x_values(domain, aggregates, config, 10)

        self.assertLessEqual(len(candidates), 20)


if __name__ == "__main__":
    unittest.main()
