"""Cost and compute estimates for LLM training experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


FLOPS_PER_PARAM_TOKEN = 6


@dataclass(frozen=True)
class CostEstimate:
    parameters: int
    tokens: int
    gpu_count: int
    gpu_tflops: float
    gpu_hourly_usd: float
    utilization: float
    overhead_multiplier: float

    @property
    def training_flops(self) -> float:
        return FLOPS_PER_PARAM_TOKEN * self.parameters * self.tokens

    @property
    def raw_gpu_hours(self) -> float:
        sustained_flops_per_hour = (
            self.gpu_count * self.gpu_tflops * 1_000_000_000_000 * 3600 * self.utilization
        )
        if sustained_flops_per_hour <= 0:
            raise ValueError("sustained FLOPs per hour must be positive")
        return self.training_flops / sustained_flops_per_hour

    @property
    def raw_cost_usd(self) -> float:
        return self.raw_gpu_hours * self.gpu_count * self.gpu_hourly_usd

    @property
    def realistic_cost_usd(self) -> float:
        return self.raw_cost_usd * self.overhead_multiplier


def estimate_from_spec(spec: dict[str, Any]) -> CostEstimate:
    if not isinstance(spec, dict):
        raise ValueError("spec must be an object")
    if spec.get("stage") not in (None, "pretrain", "midtrain", "sft", "dpo", "grpo", "distill"):
        raise ValueError("the estimator supports token-based language-model training only")
    if any(not isinstance(spec.get(key, {}), dict) for key in ("model", "resources", "budget", "training")):
        raise ValueError("model, resources, budget, and training must be objects")
    resources = spec.get("resources", {})
    if resources.get("device") in ("cpu", "mps") or resources.get("gpus", 1) == 0:
        raise ValueError("CPU/MPS workloads are unsupported by the GPU training estimator")
    for key, value in (("parameters", spec.get("model", {}).get("parameters")),
                       ("tokens_target", spec.get("training", {}).get("tokens_target")),
                       ("gpus", resources.get("gpus", 1))):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key, value, allow_zero in (
        ("gpu_tflops", resources.get("gpu_tflops", 82.0), False),
        ("gpu_hourly_usd", resources.get("gpu_hourly_usd", 0.0), True),
        ("utilization", resources.get("utilization", 0.35), False),
        ("overhead_multiplier", spec.get("budget", {}).get("overhead_multiplier", 2.0), False),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{key} must be finite and {'non-negative' if allow_zero else 'positive'}")
    if resources.get("utilization", 0.35) > 1:
        raise ValueError("utilization must not exceed 1")
    model = spec.get("model", {})
    resources = spec.get("resources", {})
    budget = spec.get("budget", {})
    training = spec.get("training", {})

    return CostEstimate(
        parameters=int(model["parameters"]),
        tokens=int(training["tokens_target"]),
        gpu_count=int(resources.get("gpus", 1)),
        gpu_tflops=float(resources.get("gpu_tflops", 82.0)),
        gpu_hourly_usd=float(resources.get("gpu_hourly_usd", 0.0)),
        utilization=float(resources.get("utilization", 0.35)),
        overhead_multiplier=float(budget.get("overhead_multiplier", 2.0)),
    )

def format_estimate(estimate: CostEstimate) -> str:
    return "\n".join(
        [
            f"parameters: {estimate.parameters}",
            f"tokens: {estimate.tokens}",
            f"training_flops: {estimate.training_flops:.3e}",
            f"raw_gpu_hours: {estimate.raw_gpu_hours:.2f}",
            f"raw_cost_usd: {estimate.raw_cost_usd:.2f}",
            f"realistic_cost_usd: {estimate.realistic_cost_usd:.2f}",
        ]
    )
