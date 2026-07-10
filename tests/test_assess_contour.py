import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import assess_contour as assess
import propose_next_sweep as sweep


class AssessContourTests(unittest.TestCase):
    def config(self):
        return sweep.ModelConfig(
            mode="monotone-y",
            monotone_direction="decreasing",
            contour_fit="local-linear",
            x_scale="linear",
            y_scale="linear",
            transition_width=0.04,
            label_noise=0.005,
            length_scale_x=0.18,
            length_scale_y=0.2,
            prior_alpha=1.0,
            prior_beta=1.0,
            grid_size=9,
            posterior_samples=0,
        )

    def test_movement_reports_grid_explicit_sse_and_grid_invariant_rmse(self):
        domain = sweep.Domain(0.0, 1.0, 0.0, 1.0)
        previous = [
            {"x": 0.0, "y_pred": 0.4},
            {"x": 0.5, "y_pred": 0.5},
            {"x": 1.0, "y_pred": 0.6},
        ]
        current = [
            {"x": 0.0, "y_pred": 0.5},
            {"x": 0.5, "y_pred": 0.6},
            {"x": 1.0, "y_pred": 0.7},
        ]

        metrics = assess.contour_movement(
            previous, current, domain, self.config(), boundary_fraction=0.2
        )

        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics["sse"], 0.03)
        self.assertAlmostEqual(metrics["rmse"], 0.1)
        self.assertAlmostEqual(metrics["edge_sse"], 0.02)

    def test_resolution_requires_brackets_and_tracks_domain_edges(self):
        domain = sweep.Domain(0.0, 1.0, 0.0, 1.0)
        aggregates = []
        for x in (0.0, 0.5, 1.0):
            aggregates.extend(
                [
                    sweep.AggregatePoint(x, 0.49, 1, 1),
                    sweep.AggregatePoint(x, 0.51, 1, 0),
                ]
            )

        metrics = assess.contour_resolution(aggregates, domain, self.config())

        self.assertEqual(metrics["bracketed_anchors"], 3)
        self.assertAlmostEqual(metrics["max_y_bracket_width"], 0.02)
        self.assertEqual(metrics["edge_bracketed"], {"lower": True, "upper": True})

    def test_generic_mode_does_not_claim_bracket_convergence(self):
        domain = sweep.Domain(0.0, 1.0, 0.0, 1.0)
        config = self.config()
        config = sweep.ModelConfig(
            mode="generic",
            monotone_direction=config.monotone_direction,
            contour_fit=config.contour_fit,
            x_scale=config.x_scale,
            y_scale=config.y_scale,
            transition_width=config.transition_width,
            label_noise=config.label_noise,
            length_scale_x=config.length_scale_x,
            length_scale_y=config.length_scale_y,
            prior_alpha=config.prior_alpha,
            prior_beta=config.prior_beta,
            grid_size=config.grid_size,
            posterior_samples=config.posterior_samples,
        )

        metrics = assess.contour_resolution([], domain, config)

        self.assertFalse(metrics["supported"])
        self.assertIn("monotone-y", metrics["reason"])

    def test_cli_persists_patience_and_converges_without_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "completed.csv"
            rows = ["caseId,x,y,id"]
            case_id = 1
            for x in (0.0, 0.5, 1.0):
                rows.append(f"{case_id},{x},0.49,1")
                case_id += 1
                rows.append(f"{case_id},{x},0.51,0")
                case_id += 1
            observations.write_text("\n".join(rows) + "\n", encoding="utf-8")
            state = root / "state.json"
            common = [
                str(observations),
                "--outfile",
                str(root / "contour.csv"),
                "--state",
                str(state),
                "--x-min",
                "0",
                "--x-max",
                "1",
                "--y-min",
                "0",
                "--y-max",
                "1",
                "--contour-points",
                "9",
                "--max-x-gap",
                "0.5",
                "--max-y-bracket-width",
                "0.03",
                "--patience",
                "1",
                "--min-iterations",
                "0",
            ]

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(assess.main([*common, "--iteration", "0"]), 0)
                self.assertEqual(assess.main([*common, "--iteration", "1"]), 0)
            payload = json.loads(state.read_text(encoding="utf-8"))

            self.assertEqual(payload["status"], "converged")
            self.assertTrue(payload["resolution_ready"])
            self.assertAlmostEqual(payload["movement"]["sse"], 0.0)


if __name__ == "__main__":
    unittest.main()
