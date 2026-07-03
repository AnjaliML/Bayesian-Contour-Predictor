import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "propose_next_sweep.py"


def synthetic_label(x, y):
    return 1 if y < 0.42 + 0.18 * x else 0


class EndToEndWorkflowTests(unittest.TestCase):
    def read_rows(self, path):
        with Path(path).open(newline="") as handle:
            return list(csv.DictReader(handle))

    def write_rows(self, path, rows):
        with Path(path).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["caseId", "x", "y", "id"])
            writer.writeheader()
            writer.writerows(rows)

    def test_two_sweep_campaign_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            sweep0 = Path(directory) / "Sweep-0.csv"
            initial_rows = [
                {"caseId": "1", "x": "0.0", "y": "0.20", "id": "1"},
                {"caseId": "2", "x": "0.0", "y": "0.75", "id": "0"},
                {"caseId": "3", "x": "0.5", "y": "0.30", "id": "1"},
                {"caseId": "4", "x": "0.5", "y": "0.70", "id": "0"},
                {"caseId": "5", "x": "1.0", "y": "0.35", "id": "1"},
                {"caseId": "6", "x": "1.0", "y": "0.85", "id": "0"},
            ]
            self.write_rows(sweep0, initial_rows)

            sweep1_proposed = Path(directory) / "Sweep-1.csv"
            command = [
                sys.executable,
                str(SCRIPT),
                str(sweep0),
                "--outfile",
                str(sweep1_proposed),
                "--n-simulations",
                "6",
                "--seed",
                "21",
                "--x-min",
                "0",
                "--x-max",
                "1",
                "--y-min",
                "0.1",
                "--y-max",
                "0.9",
                "--grid-size",
                "9",
                "--posterior-samples",
                "4",
            ]
            subprocess.run(command, check=True)
            proposed_rows = self.read_rows(sweep1_proposed)
            self.assertEqual(len(proposed_rows), 6)
            self.assertEqual({row["id"] for row in proposed_rows}, {"-1"})

            completed_rows = []
            for row in proposed_rows:
                x = float(row["x"])
                y = float(row["y"])
                completed_rows.append(
                    {
                        "caseId": row["caseId"],
                        "x": row["x"],
                        "y": row["y"],
                        "id": str(synthetic_label(x, y)),
                    }
                )
            sweep1_completed = Path(directory) / "Sweep-1-completed.csv"
            self.write_rows(sweep1_completed, completed_rows)

            sweep2_proposed = Path(directory) / "Sweep-2.csv"
            subprocess.run(
                [
                    *command[:3],
                    str(sweep1_completed),
                    "--outfile",
                    str(sweep2_proposed),
                    *command[5:],
                ],
                check=True,
            )
            next_rows = self.read_rows(sweep2_proposed)

            self.assertEqual(len(next_rows), 6)
            self.assertEqual({row["id"] for row in next_rows}, {"-1"})
            self.assertTrue(any(row["proposal_type"] == "repeat" for row in next_rows))
            self.assertTrue(any(row["proposal_type"] == "new" for row in next_rows))
            self.assertTrue(all(row["y_c_std"] for row in next_rows))


if __name__ == "__main__":
    unittest.main()
