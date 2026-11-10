# compute-economics

Last updated: 2026-11-10

Estimate workload compute and cost from explicit caller-supplied assumptions.
This standard-library Python 3.11+ package can be used independently of
compute-planner, compute-runners, Lab, GPUs, accounts, and network services.

**Status: implemented primitives, version 0.1.0.** The current model estimates
token-based language-model GPU training with the 6NT compute approximation.
It is not a general cloud billing engine or a live pricing catalogue.

## Independent use

From this repository:

```sh
python3 -m pip install .
compute-economics estimate-cost examples/estimate.json
python3 -m compute_economics.cli estimate-cost examples/estimate.json
python3 examples/quickstart.py
```

```python
from compute_economics import CostEstimate, estimate_from_spec, format_estimate
```

`estimate_from_spec` takes a model parameter count, token target, and explicit
resource/budget assumptions. It returns a `CostEstimate` with training FLOPs,
`wall_clock_hours` (elapsed time), `aggregate_gpu_hours` (sum across GPUs),
raw cost, and overhead-adjusted cost. Cost is aggregate GPU-hours multiplied
by the supplied hourly GPU rate. Example rates are authored
fixture inputs, not current provider quotes. CPU/MPS workloads, serving stages,
and non-finite resource values are rejected.

`resources.gpu_hourly_usd` must be supplied explicitly. A missing rate raises
an error; zero is accepted only when the caller explicitly supplies a free-rate
assumption. Other model defaults are one GPU, 82 GPU TFLOPs, 0.35 utilization,
and a 2.0 overhead multiplier; these are configurable assumptions, not measured
hardware characteristics.

The approximation assumes identical GPUs and linear scaling: adding GPUs
reduces elapsed time while conserving aggregate GPU-hours. Communication,
storage, teacher inference, rollout generation, and actual scaling efficiency
are not independently modeled.

## Ownership and composition

This component owns estimates, their assumptions, and the cost-ledger template.
A scheduler can consume its output through ordinary values such as
`Job.expected_cost_usd`; no compute-planner import or mandatory dependency is
required. The estimator does not schedule, route, allocate resources, enforce
budgets, or measure actual invoices.

[Specification](docs/specs/README.md) ·
[Current provenance](docs/sources/extraction-v2.json) ·
[Correction and source checks](docs/results/2026/11-10/economics-correction-v2.json) ·
[Original extraction snapshot](docs/sources/extraction.json)

The original implementation came from compute-planner 0.1.0. Its first commit
here preserves the moved bytes; packaging adapted imports, and a separate
reviewed correction then made prices explicit and clarified hour units. The
current implementation is therefore not byte-identical to the inherited one.
The former `raw_gpu_hours` property/format field was replaced by the two
explicit hour measures above. No compatibility alias remains.

No fresh install or wheel build is claimed by the source checks. MIT applies
to the authored code and documentation; third-party inputs retain their terms.
