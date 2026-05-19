# Safety_Drone — Sweep Results

**Property**: `relative_confirmed -> (TRUE S velocity_sent)`

**Sweep**: 15 explicit runs over trace sizes {1k, 10k, 100k} × ε ∈ {0.0, 0.5, 1.0, 5.0, inf}
**Timeout**: 1000s per run (TIMEOUT entries indicate the run was killed)

| Exp | Trace | ε | Verdict | Nodes | Removed | MaxSum | Events | Time (s) | Mem (MB) | CPU |
|-----|-------|---|---------|-------|---------|--------|--------|----------|----------|-----|
| drone | 1k | 0.0 | SAT | 41 | 950 | 1 | 990 | 0.19 | 22.9 | 92% |
| drone | 1k | 0.5 | SAT | 41 | 950 | 1 | 990 | 0.17 | 22.9 | 98% |
| drone | 1k | 1.0 | SAT | 41 | 950 | 1 | 990 | 0.18 | 22.9 | 98% |
| drone | 1k | 5.0 | SAT | 41 | 950 | 1 | 990 | 0.18 | 22.9 | 97% |
| drone | 1k | inf | SAT | 41 | 950 | 1 | 990 | 0.17 | 22.9 | 98% |
| drone | 10k | 0.0 | SAT | 40 | 9951 | 1 | 9990 | 1.38 | 40.7 | 99% |
| drone | 10k | 0.5 | SAT | 40 | 9951 | 1 | 9990 | 1.38 | 40.7 | 99% |
| drone | 10k | 1.0 | SAT | 40 | 9951 | 1 | 9990 | 1.42 | 40.7 | 99% |
| drone | 10k | 5.0 | SAT | 40 | 9951 | 1 | 9990 | 1.37 | 40.4 | 99% |
| drone | 10k | inf | SAT | 40 | 9951 | 1 | 9990 | 1.39 | 40.8 | 99% |
| drone | 100k | 0.0 | SAT | 44 | 99947 | 1 | 99990 | 13.51 | 227.7 | 99% |
| drone | 100k | 0.5 | SAT | 44 | 99947 | 1 | 99990 | 14.49 | 216.6 | 96% |
| drone | 100k | 1.0 | SAT | 44 | 99947 | 1 | 99990 | 14.31 | 228.1 | 97% |
| drone | 100k | 5.0 | SAT | 44 | 99947 | 1 | 99990 | 15.11 | 210.5 | 93% |
| drone | 100k | inf | SAT | 44 | 99947 | 1 | 99990 | 14.02 | 227.5 | 98% |

_Generated 2026-05-05 16:08:19_
