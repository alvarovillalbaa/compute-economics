import unittest

from compute_economics.costing import estimate_from_spec


class CostingTests(unittest.TestCase):
    def test_unsupported_and_nonfinite_inputs_fail(self):
        baseline = {"model": {"parameters": 10}, "training": {"tokens_target": 100},
                    "resources": {"gpus": 1, "gpu_hourly_usd": 1.0}}
        for change in ({"stage": "serve"}, {"resources": {"device": "cpu", "gpus": 0, "gpu_hourly_usd": 1.0}},
                       {"resources": {"gpus": 1, "gpu_hourly_usd": 1.0, "utilization": float("nan")}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                estimate_from_spec({**baseline, **change})

    def test_estimate_from_spec_has_positive_flops(self):
        spec = {
            "model": {"parameters": 124_000_000},
            "training": {"tokens_target": 500_000_000},
            "resources": {
                "gpus": 1,
                "gpu_tflops": 82,
                "gpu_hourly_usd": 0.69,
                "utilization": 0.35,
            },
            "budget": {"overhead_multiplier": 2.0},
        }

        estimate = estimate_from_spec(spec)

        self.assertEqual(estimate.training_flops, 6 * 124_000_000 * 500_000_000)
        self.assertAlmostEqual(estimate.raw_cost_usd, estimate.aggregate_gpu_hours * 0.69)
        self.assertGreater(estimate.wall_clock_hours, 0)
        self.assertGreater(estimate.aggregate_gpu_hours, 0)
        self.assertGreater(estimate.realistic_cost_usd, estimate.raw_cost_usd)

    def test_missing_price_is_not_inferred_to_be_free(self):
        spec = {"model": {"parameters": 10}, "training": {"tokens_target": 100},
                "resources": {"gpus": 1}}
        with self.assertRaisesRegex(ValueError, "gpu_hourly_usd must be explicitly supplied"):
            estimate_from_spec(spec)

    def test_free_price_requires_explicit_zero(self):
        spec = {"model": {"parameters": 10}, "training": {"tokens_target": 100},
                "resources": {"gpus": 1, "gpu_hourly_usd": 0}}
        estimate = estimate_from_spec(spec)
        self.assertGreater(estimate.wall_clock_hours, 0)
        self.assertEqual(estimate.raw_cost_usd, 0)
        self.assertEqual(estimate.realistic_cost_usd, 0)

    def test_multi_gpu_hours_and_cost_conserve_work_under_linear_scaling(self):
        spec = {"model": {"parameters": 124_000_000},
                "training": {"tokens_target": 500_000_000},
                "resources": {"gpus": 1, "gpu_tflops": 82,
                              "gpu_hourly_usd": 0.69, "utilization": 0.35},
                "budget": {"overhead_multiplier": 2.0}}
        single = estimate_from_spec(spec)
        multi = estimate_from_spec({**spec, "resources": {**spec["resources"], "gpus": 8}})
        self.assertAlmostEqual(multi.wall_clock_hours, single.wall_clock_hours / 8)
        self.assertAlmostEqual(multi.aggregate_gpu_hours, multi.wall_clock_hours * 8)
        self.assertAlmostEqual(multi.aggregate_gpu_hours, single.aggregate_gpu_hours)
        self.assertAlmostEqual(multi.raw_cost_usd, single.raw_cost_usd)
        self.assertAlmostEqual(multi.realistic_cost_usd, single.realistic_cost_usd)

    def test_hour_units_for_one_teraflop_gpu_hour_of_work(self):
        spec = {"model": {"parameters": 600_000_000},
                "training": {"tokens_target": 1_000_000},
                "resources": {"gpus": 4, "gpu_tflops": 1,
                              "gpu_hourly_usd": 2.0, "utilization": 1.0},
                "budget": {"overhead_multiplier": 1.0}}
        estimate = estimate_from_spec(spec)
        self.assertEqual(estimate.training_flops, 3.6e15)
        self.assertAlmostEqual(estimate.wall_clock_hours, 0.25)
        self.assertAlmostEqual(estimate.aggregate_gpu_hours, 1.0)
        self.assertAlmostEqual(estimate.raw_cost_usd, 2.0)


if __name__ == "__main__":
    unittest.main()
