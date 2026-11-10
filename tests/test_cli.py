import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import unittest

from compute_economics.cli import main


class CliTests(unittest.TestCase):
    def test_native_estimate_example_runs_without_planner(self):
        example = Path(__file__).resolve().parents[1] / "examples/estimate.json"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["estimate-cost", str(example)]), 0)
        self.assertIn("training_flops: 3.720e+17", stdout.getvalue())
        self.assertIn("wall_clock_hours:", stdout.getvalue())
        self.assertIn("aggregate_gpu_hours:", stdout.getvalue())
        self.assertNotIn("raw_gpu_hours:", stdout.getvalue())
        self.assertIn("realistic_cost_usd:", stdout.getvalue())

    def test_unreadable_input_reports_failure(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(main(["estimate-cost", "nonexistent-economics-fixture.json"]), 1)
        self.assertIn("ERROR:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
