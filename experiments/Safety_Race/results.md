# Safety_Race — Sweep Results

**Property**: `!(TRUE S unsafe_A1A2) & !(TRUE S unsafe_A1A3) & !(TRUE S unsafe_A2A3)`

**Sweep**: 15 explicit runs over trace sizes {1k, 10k, 100k} × ε ∈ {0.0, 0.5, 1.0, 5.0, inf}
**Timeout**: 1000s per run (TIMEOUT entries indicate the run was killed)

| Exp | Trace | ε | Verdict | Nodes | Removed | MaxSum | Events | Time (s) | Mem (MB) | CPU |
|-----|-------|---|---------|-------|---------|--------|--------|----------|----------|-----|
| race | 1k | 0.0 | SAT | 31 | 6139 | 1 | 995 | 2.32 | 23.4 | 99% |
| race | 1k | 0.5 | SAT | 31 | 6139 | 1 | 995 | 2.28 | 23.2 | 99% |
| race | 1k | 1.0 | SAT | 170 | 36492 | 1 | 995 | 14.86 | 25.3 | 98% |
| race | 1k | 5.0 | SAT | 3714 | 851648 | 1 | 995 | 375.55 | 79.1 | 99% |
| race | 1k | inf | ABORTED | - | - | - | - | (user) | ? | ? |
| race | 10k | 0.0 | SAT | 31 | 61939 | 1 | 9995 | 21.86 | 41.2 | 99% |
| race | 10k | 0.5 | SAT | 31 | 61939 | 1 | 9995 | 21.87 | 41.1 | 99% |
| race | 10k | 1.0 | SAT | 170 | 369492 | 1 | 9995 | 145.75 | 43.1 | 99% |
| race | 10k | 5.0 | TIMEOUT | - | - | - | - | >1000 | 96.1 | 99% |
| race | 10k | inf | ABORTED | - | - | - | - | (user) | ? | ? |
| race | 100k | 0.0 | SAT | 31 | 619939 | 1 | 99995 | 218.66 | 222.1 | 99% |
| race | 100k | 0.5 | SAT | 31 | 619939 | 1 | 99995 | 221.03 | 224.5 | 99% |
| race | 100k | 1.0 | ABORTED | - | - | - | - | (user) | ? | ? |
| race | 100k | 5.0 | ABORTED | - | - | - | - | (user) | ? | ? |
| race | 100k | inf | ABORTED | - | - | - | - | (user) | ? | ? |

_Generated 2026-04-26 23:21:06_
