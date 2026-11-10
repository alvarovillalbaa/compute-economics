# Compute economics specification

Last updated: 2026-11-10

Version 0.1.0 provides the extracted language-model training estimate model and
an estimate-cost CLI. Inputs are explicit caller assumptions; outputs are
estimates rather than measurements or billing guarantees.

## Implemented interface

- `CostEstimate`: parameters, tokens, GPU count/FLOPs/price, utilization, overhead.
- `estimate_from_spec(spec)`: validate the supported narrow workload and produce an estimate.
- `format_estimate(estimate)`: readable FLOP/hour/cost output.
- `compute-economics estimate-cost SPEC.json`: the same calculation from JSON.

The model computes approximately 6 × parameters × tokens training FLOPs.
Resource assumptions determine GPU hours and cost. Native examples and tests
define the extracted behavior; mathematical precision does not establish
hardware performance or provider pricing.

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

Native source checks are recorded separately from fresh-install and wheel
proof. Those installation/build checks remain pending for this extraction.
