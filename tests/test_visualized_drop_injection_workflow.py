import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "end-to-end-with-visualization-drop-injection.py"


class VisualizedDropInjectionWorkflowTests(unittest.TestCase):
    def read_rows(self, path):
        with Path(path).open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_visualizer_runs_one_complete_campaign_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "visualizer-run"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--iterations",
                    "1",
                    "--initial-points",
                    "4",
                    "--batch-size",
                    "3",
                    "--preview-points",
                    "4",
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
                ],
                check=True,
            )

            state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
            proposed_rows = self.read_rows(output_dir / "Sweep-1_proposed.csv")
            completed_rows = self.read_rows(output_dir / "Sweep-1_completed.csv")

            self.assertTrue((output_dir / "index.html").exists())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["iteration"], 1)
            self.assertEqual(len(proposed_rows), 3)
            self.assertEqual(len(completed_rows), 3)
            self.assertEqual(len(state["completed"]), 7)
            self.assertEqual({row["id"] for row in completed_rows}, {"0", "1"})
            self.assertTrue(state["true_contour"])
            self.assertTrue(state["contour"])


if __name__ == "__main__":
    unittest.main()
