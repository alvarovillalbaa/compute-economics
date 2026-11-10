from compute_economics import estimate_from_spec, format_estimate

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
print(format_estimate(estimate_from_spec(spec)))
