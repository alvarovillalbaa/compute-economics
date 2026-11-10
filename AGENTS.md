# compute-economics agent instructions

- Read README.md and docs/specs/README.md before changing behavior.
- This repository owns workload cost estimates. Never import lab, compute-planner, or assume sibling checkouts.
- Estimates use explicit caller assumptions; never silently fetch live prices or treat estimates as observed billing.
- Preserve original-source provenance and historical evidence. Keep changes purpose grouped.
- Do not install GPU stacks, contact providers, allocate resources, or enforce budgets implicitly.
- Keep credentials and generated artifacts out of Git. Run focused source checks and git diff --check.
