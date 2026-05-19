# Safety_Cassandra — Sweep Results

**Property**: `write_enrollment -> (TRUE S write_student)`

**Sweep**: 15 explicit runs over trace sizes {1k, 10k, 100k} × ε ∈ {0.0, 0.5, 1.0, 5.0, inf}
**Timeout**: 1000s per run (TIMEOUT entries indicate the run was killed)

| Exp | Trace | ε | Verdict | Nodes | Removed | MaxSum | Events | Time (s) | Mem (MB) | CPU |
|-----|-------|---|---------|-------|---------|--------|--------|----------|----------|-----|
| cassandra | 1k | 0.0 | SAT | 5 | 991 | 1 | 995 | 0.14 | 22.4 | 87% |
| cassandra | 1k | 0.5 | SAT | 9 | 995 | 1 | 995 | 0.12 | 22.5 | 97% |
| cassandra | 1k | 1.0 | SAT | 17 | 1001 | 1 | 995 | 0.12 | 22.5 | 97% |
| cassandra | 1k | 5.0 | SAT | 120 | 1041 | 1 | 995 | 0.13 | 22.9 | 96% |
| cassandra | 1k | inf | SAT | 236 | 1087 | 1 | 995 | 0.13 | 23.2 | 97% |
| cassandra | 10k | 0.0 | SAT | 5 | 9991 | 1 | 9995 | 0.66 | 35.6 | 99% |
| cassandra | 10k | 0.5 | SAT | 9 | 10002 | 1 | 9995 | 0.66 | 35.6 | 99% |
| cassandra | 10k | 1.0 | SAT | 17 | 10022 | 1 | 9995 | 0.66 | 36.0 | 99% |
| cassandra | 10k | 5.0 | SAT | 437 | 10370 | 1 | 9995 | 0.71 | 37.7 | 99% |
| cassandra | 10k | inf | SAT | 838 | 10631 | 1 | 9995 | 0.71 | 39.3 | 99% |
| cassandra | 100k | 0.0 | SAT | 14 | 99982 | 1 | 99995 | 6.41 | 179.6 | 97% |
| cassandra | 100k | 0.5 | SAT | 14 | 99982 | 1 | 99995 | 6.17 | 180.5 | 99% |
| cassandra | 100k | 1.0 | SAT | 14 | 99982 | 1 | 99995 | 6.40 | 172.4 | 94% |
| cassandra | 100k | 5.0 | SAT | 14 | 99982 | 1 | 99995 | 6.28 | 180.0 | 99% |
| cassandra | 100k | inf | SAT | 14 | 99982 | 1 | 99995 | 6.26 | 179.0 | 98% |

_Generated 2026-04-26 23:24:21_
