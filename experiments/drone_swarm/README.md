# Drone Swarm Scalability Experiments

## Overview

Four independent experiments evaluating PROVE's runtime verification performance
on 3-drone swarm traces. Each experiment tests a different EPLTL property with
its own traces (1K, 5K, 10K, 100K events), logs, and results.

## Prerequisites

```bash
brew install gnu-time coreutils
source .venv/bin/activate
pip install -e ".[dev]"
```

## Experiments

| # | Name | Formula | Operators | Expected |
|---|------|---------|-----------|----------|
| 1 | Fragile Ordering | `late_beacon -> @early_beacon` | Implication + Yesterday | SAT when eps >= 2.0, VIOL when eps < 2.0 |
| 2 | Mutual Exclusion | `!(TRUE S (idle & scanning))` | Negation + Since + Conjunction | Always VIOL |
| 3 | Continuous Since | `relative_computed -> (position_sent S position_measured)` | Implication + Since | Always SAT |
| 4 | Phase Transition | `scanning <-> @idle` | Biconditional + Yesterday | Always SAT |

## Running

Each experiment is self-contained. Run from the project root:

```bash
# Run one experiment
bash experiments/drone_swarm/exp1_fragile_ordering/run_experiment.sh

# Run all four sequentially
for exp in exp1_fragile_ordering exp2_mutual_exclusion exp3_continuous_since exp4_phase_transition; do
    bash experiments/drone_swarm/$exp/run_experiment.sh
done
```

Each script will:
1. Generate traces (1K, 5K, 10K, 100K) if they don't already exist
2. Validate traces
3. Run PROVE for each trace x epsilon (0.0, 0.5, 1.0, 2.0, 5.0, inf) = 24 runs
4. Kill any run exceeding 1000 seconds (marked as TIMEOUT)
5. Save per-run logs + summary.log + results.md

## Output

Each experiment folder contains after running:

```
exp{N}_{name}/
├── run_experiment.sh          # Self-contained runner
├── {name}.prop                # EPLTL property
├── traces/                    # Generated traces (1K, 5K, 10K, 100K)
├── logs/
│   ├── summary.log            # All runs summary
│   ├── 1K_eps0.0.log          # Per-run logs
│   ├── 1K_eps0.5.log
│   └── ...
└── results.md                 # Markdown results table
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Drones | 3 (Drone1, Drone2, Drone3) |
| Message rate | 30% |
| Fragile delta | 2.0 |
| Seed | 42 |
| Trace sizes | 1K, 5K, 10K, 100K |
| Epsilons | 0.0, 0.5, 1.0, 2.0, 5.0, inf |
| Timeout | 1000s per run |
| Runs per experiment | 24 |
| Total runs | 96 |
