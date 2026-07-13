import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "rearm_campaign.sh"


class RearmCampaignTests(unittest.TestCase):
    def test_campaign_resumes_without_overwriting_completed_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign"
            completed = campaign / "completed"
            completed.mkdir(parents=True)
            (completed / "Sweep-0_completed.csv").write_text(
                "\n".join(
                    [
                        "caseId,x,y,id",
                        "1,1,0.004,1",
                        "2,1,0.04,0",
                        "3,100,0.01,1",
                        "4,100,0.06,0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            runner = root / "runner.py"
            runner.write_text(
                """#!/usr/bin/env python3
import csv,sys
with open(sys.argv[1], newline='') as source, open(sys.argv[2], 'w', newline='') as target:
    rows = list(csv.DictReader(source))
    writer = csv.DictWriter(target, fieldnames=['caseId', 'x', 'y', 'id'])
    writer.writeheader()
    for row in rows:
        writer.writerow({'caseId': row['caseId'], 'x': row['x'], 'y': row['y'], 'id': int(float(row['y']) < 0.02)})
""",
                encoding="utf-8",
            )
            runner.chmod(0o755)
            environment = {
                **os.environ,
                "CAMPAIGN_DIR": str(campaign),
                "BATCH_RUNNER": str(runner),
                "BATCH_SIZE": "2",
                "MAX_ITERATIONS": "1",
                "X_CANDIDATES": "1,100",
            }

            subprocess.run(["bash", str(SCRIPT)], check=True, env=environment, capture_output=True)
            first_batch = (completed / "Sweep-1_completed.csv").read_text(encoding="utf-8")

            environment["MAX_ITERATIONS"] = "2"
            subprocess.run(["bash", str(SCRIPT)], check=True, env=environment, capture_output=True)
            state = json.loads((campaign / "state.json").read_text(encoding="utf-8"))

            self.assertEqual(
                (completed / "Sweep-1_completed.csv").read_text(encoding="utf-8"),
                first_batch,
            )
            self.assertTrue((completed / "Sweep-2_completed.csv").exists())
            self.assertEqual(state["iteration"], 2)
            with (completed / "Sweep-2_completed.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertLessEqual({row["id"] for row in rows}, {"0", "1"})
            self.assertLessEqual({float(row["x"]) for row in rows}, {1.0, 100.0})


if __name__ == "__main__":
    unittest.main()
