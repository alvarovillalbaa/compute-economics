# Compute economics specification

Last updated: 2026-11-10

Version 0.1.0 provides the extracted language-model training estimate model and
an estimate-cost CLI. Inputs are explicit caller assumptions; outputs are
estimates rather than measurements or billing guarantees.

## Implemented interface

- `CostEstimate`: parameters, tokens, GPU count/FLOPs/price, utilization, overhead;
  properties expose `wall_clock_hours`, `aggregate_gpu_hours`, and cost.
- `estimate_from_spec(spec)`: validate the supported narrow workload and produce an estimate.
- `format_estimate(estimate)`: readable FLOP/hour/cost output.
- `compute-economics estimate-cost SPEC.json`: the same calculation from JSON.

The model computes approximately 6 × parameters × tokens training FLOPs.
Resource assumptions determine elapsed hours, aggregate GPU-hours, and cost.
The hourly GPU rate is mandatory: omission fails rather than implying zero.
An explicitly supplied zero rate is allowed as a caller's free assumption.
The former ambiguous `raw_gpu_hours` property and output field are removed.

For fixed work and identical GPUs, the approximation assumes linear scaling:
elapsed hours decrease with GPU count, aggregate GPU-hours remain constant, and
cost uses aggregate GPU-hours × rate. This does not model measured scaling
efficiency, communication, storage, teacher inference, or rollout costs.
Native examples and tests
define the extracted behavior; mathematical precision does not establish
hardware performance or provider pricing.

Use `estimate_from_spec` for input validation; `CostEstimate` itself is a direct
data container. GPU count, TFLOPs, utilization, and overhead have documented
defaults, while price must always be explicit.

## Boundary

Scheduler priorities, placement, provider capability matching, and execution
remain outside this component. Planning can use this package or another
estimator without adopting its implementation. A cost ledger records facts
entered by a caller and is not an invoice or telemetry collection system.

## Future work

- Compare estimates against measured workloads with explicit methodology.
- Add workload models only with declared supported inputs and limitations.
- Keep any future price snapshots versioned with source, currency, units, and time.
- Add pricing adapters only when needed; do not hide provider calls in pure estimates.

The [v2 correction receipt](../results/2026/11-10/economics-correction-v2.json)
records the reviewed price/unit correction and its tests. The original extraction
receipt remains historical evidence. Source checks are separate from fresh-install
and wheel proof, which the integration owner runs centrally.
