import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import classify_drops_sized_based as sized


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "end-to-end-with-visualization-drop-injection.py"


class VisualizedDropInjectionWorkflowTests(unittest.TestCase):
    def read_rows(self, path):
        with Path(path).open(newline="") as handle:
            return list(csv.DictReader(handle))

    def run_visualizer(self, args):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            self.fail(
                "visualizer failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return result

    def test_visualizer_runs_one_complete_campaign_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "visualizer-run"
            result = self.run_visualizer(
                [
                    "--iterations",
                    "1",
                    "--initial-points",
                    "4",
                    "--batch-size",
                    "3",
                    "--n-repeats",
                    "0",
                    "--preview-points",
                    "4",
                    "--preview-grid-size",
                    "5",
                    "--grid-size",
                    "5",
                    "--posterior-samples",
                    "1",
                    "--delay",
                    "0",
                    "--no-server",
                    "--no-browser",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
            html = (output_dir / "index.html").read_text(encoding="utf-8")
            proposed_rows = self.read_rows(output_dir / "Sweep-1_proposed.csv")
            completed_rows = self.read_rows(output_dir / "Sweep-1_completed.csv")

            self.assertTrue((output_dir / "index.html").exists())
            self.assertTrue((output_dir / "Sweep-1_contour-final.csv").exists())
            self.assertIn("RUNNING - not converged yet", html)
            self.assertIn("CONVERGED", html)
            self.assertIn("COMPLETED", html)
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["iteration"], 1)
            self.assertEqual(state["total_iterations"], 1)
            self.assertEqual(state["batch_size"], 3)
            self.assertEqual(state["domain"]["rr_min"], 1.0)
            self.assertEqual(state["domain"]["rr_max"], 100.0)
            self.assertEqual(state["domain"]["rr_scale"], "log10")
            self.assertEqual(state["domain"]["oh_min"], 0.001)
            self.assertEqual(state["domain"]["oh_max"], 0.1)
            self.assertEqual(state["domain"]["oh_scale"], "log10")
            self.assertEqual(len(proposed_rows), 3)
            self.assertEqual(len(completed_rows), 3)
            self.assertEqual(len(state["completed"]), 7)
            self.assertLessEqual({row["id"] for row in completed_rows}, {"0", "1"})
            self.assertEqual(len(state["history"]), 1)
            self.assertEqual(state["history"][0]["completed"], 3)
            self.assertTrue(state["true_contour"])
            self.assertTrue(state["contour"])
            self.assertLessEqual(len(state["messages"]), 16)
            self.assertTrue(any("proposing 3 runs" in item for item in state["messages"]))
            self.assertTrue(
                any("running 3 simulated experiments" in item for item in state["messages"])
            )
            self.assertIn("Sweep 1: proposing 3 runs", result.stdout)
            self.assertIn("Sweep 1: running 3 simulated experiments", result.stdout)

    def test_visualizer_can_stop_after_contour_convergence(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "visualizer-converged"
            self.run_visualizer(
                [
                    "--iterations",
                    "5",
                    "--initial-points",
                    "4",
                    "--batch-size",
                    "3",
                    "--n-repeats",
                    "0",
                    "--preview-points",
                    "4",
                    "--preview-grid-size",
                    "5",
                    "--preview-every",
                    "1",
                    "--grid-size",
                    "5",
                    "--posterior-samples",
                    "0",
                    "--convergence-rms-tolerance",
                    "999",
                    "--convergence-max-tolerance",
                    "999",
                    "--convergence-boundary-tolerance",
                    "999",
                    "--convergence-max-x-gap",
                    "1",
                    "--convergence-max-y-bracket-width",
                    "1",
                    "--convergence-allow-unbracketed-edges",
                    "--convergence-patience",
                    "1",
                    "--convergence-min-iterations",
                    "1",
                    "--delay",
                    "0",
                    "--no-server",
                    "--no-browser",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))

            self.assertEqual(state["status"], "converged")
            self.assertLess(state["iteration"], state["total_iterations"])
            self.assertTrue(state["contour"])

    def test_visualizer_can_use_size_based_classifier(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "visualizer-sized"
            self.run_visualizer(
                [
                    "--classifier",
                    "size-based",
                    "--iterations",
                    "1",
                    "--initial-points",
                    "4",
                    "--batch-size",
                    "3",
                    "--n-repeats",
                    "0",
                    "--preview-points",
                    "4",
                    "--preview-grid-size",
                    "5",
                    "--grid-size",
                    "5",
                    "--posterior-samples",
                    "0",
                    "--delay",
                    "0",
                    "--no-server",
                    "--no-browser",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
            first_true = state["true_contour"][0]

            self.assertEqual(state["status"], "complete")
            self.assertAlmostEqual(
                first_true["Oh"],
                sized.size_threshold_y(first_true["Rr"]),
            )
            self.assertEqual(len(self.read_rows(output_dir / "Sweep-1_completed.csv")), 3)


if __name__ == "__main__":
    unittest.main()
