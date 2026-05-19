# Safety_Drone (by drone count) — Sweep Results

**Property**: `relative_confirmed -> (TRUE S velocity_sent)`

**Sweep**: 15 explicit runs over drone counts K ∈ {2, 4, 6, 8, 10} × trace sizes {1k, 10k, 100k}, all at fixed ε = 1.0.
**Timeout**: 1000s per run (TIMEOUT entries indicate the run was killed)

| Exp | K (drones) | Trace | ε | Verdict | Nodes | Removed | MaxSum | Events | Time (s) | Mem (MB) | CPU |
|-----|------------|-------|---|---------|-------|---------|--------|--------|----------|----------|-----|
| drone | 2 | 1k | 1.0 | SAT | 4 | 995 | 1 | 998 | 0.12 | 22.3 | 84% |
| drone | 2 | 10k | 1.0 | SAT | 4 | 9995 | 1 | 9998 | 0.41 | 33.7 | 98% |
| drone | 2 | 100k | 1.0 | SAT | 4 | 99995 | 1 | 99998 | 3.71 | 157.8 | 99% |
| drone | 4 | 1k | 1.0 | SAT | 11 | 986 | 1 | 996 | 0.11 | 22.4 | 96% |
| drone | 4 | 10k | 1.0 | SAT | 14 | 9983 | 1 | 9996 | 0.57 | 35.2 | 98% |
| drone | 4 | 100k | 1.0 | SAT | 14 | 99983 | 1 | 99996 | 5.32 | 170.7 | 99% |
| drone | 6 | 1k | 1.0 | SAT | 23 | 972 | 1 | 994 | 0.14 | 22.8 | 96% |
| drone | 6 | 10k | 1.0 | SAT | 22 | 9973 | 1 | 9994 | 0.77 | 39.5 | 99% |
| drone | 6 | 100k | 1.0 | SAT | 20 | 99975 | 1 | 99994 | 7.40 | 208.5 | 99% |
| drone | 8 | 1k | 1.0 | SAT | 32 | 961 | 1 | 992 | 0.16 | 22.8 | 97% |
| drone | 8 | 10k | 1.0 | SAT | 32 | 9961 | 1 | 9992 | 1.05 | 39.8 | 99% |
| drone | 8 | 100k | 1.0 | SAT | 32 | 99961 | 1 | 99992 | 9.71 | 219.7 | 99% |
| drone | 10 | 1k | 1.0 | SAT | 41 | 950 | 1 | 990 | 0.20 | 22.9 | 93% |
| drone | 10 | 10k | 1.0 | SAT | 40 | 9951 | 1 | 9990 | 1.37 | 40.7 | 99% |
| drone | 10 | 100k | 1.0 | SAT | 44 | 99947 | 1 | 99990 | 13.10 | 222.3 | 99% |

_Generated 2026-04-27 07:16:47_
