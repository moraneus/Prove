# PROVE -- Partial oRder Verification Engine

Runtime verification of Existential Past Linear Temporal Logic (EPLTL) properties over partial order executions of distributed systems.

## Overview

PROVE implements a sliding window algorithm for checking whether **at least one valid linearization** of a partial order execution satisfies a given past-time temporal property. It combines:

- **Fidge-Mattern vector clocks** for causal ordering
- **Bounded clock skew (epsilon)** for additional temporal ordering
- **Sliding window graph** for efficient state space management
- **Summary-based evaluation** for EPLTL formula checking

Based on the paper: *"Runtime Verification of Linear Temporal Properties over Partial Order Executions"* by Doron Peled et al.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/moraneus/Prove.git
cd Prove
pip install -e ".[dev]"
```

## Quick Start

```bash
# Check a property against a trace
prove -p examples/properties/safety.prop -t examples/traces/simple.csv

# With epsilon (clock skew bound)
prove -p examples/properties/response.prop -t examples/traces/message_passing.csv -e 2.0

# Verbose output with ASCII visualization
prove -p examples/properties/safety.prop -t examples/traces/paper_example.csv \
      -e 2.0 -o verbose --visualize-ascii --stats
```

## CLI Reference

```
prove -p PROPERTY -t TRACE [options]
```

| Option | Description |
|--------|-------------|
| `-p, --property FILE` | EPLTL formula file (required) |
| `-t, --trace FILE` | Trace CSV file (required) |
| `-e, --epsilon N` | Maximum clock skew in time units (default: infinity) |
| `-o, --output MODE` | Output level: `silent`, `normal`, `verbose` (default: normal) |
| `-d, --debug N` | Debug level 0-3 (default: 0) |
| `--visualize [FILE]` | Generate graph visualization (DOT format) |
| `--visualize-ascii` | Print ASCII partial order diagram |
| `--full-graph` | Keep all graph nodes (disable pruning) |
| `--stats` | Print verification statistics |
| `--version` | Show version and exit |

**Exit codes:** `0` = property satisfied, `1` = property violated, `2` = error

## Trace File Format

CSV with one event per row:

```csv
# system_processes: P1|P2
# epsilon: 2.0
eid,processes,vc,timestamp,props,event_type,msg_partner
iota_P1,P1,P1:1;P2:0,0.0,init,local,
iota_P2,P2,P1:0;P2:1,0.0,init,local,
e1,P1,P1:2;P2:0,1.0,ready,local,
e2,P2,P1:0;P2:2,1.5,done,local,
```

| Field | Description |
|-------|-------------|
| `eid` | Unique event identifier |
| `processes` | Process ID this event belongs to |
| `vc` | Vector clock (`P1:2;P2:1`) |
| `timestamp` | Global timestamp for epsilon-ordering |
| `props` | Propositions (`ready\|done`), may be empty |
| `event_type` | `local`, `send`, or `receive` |
| `msg_partner` | Target (send) or source (receive) process |

## Property File Format (EPLTL)

One formula per file. Lines starting with `#` are comments.

| Operator | Syntax | Description |
|----------|--------|-------------|
| True | `TRUE`, `true` | Always true |
| False | `FALSE`, `false` | Always false |
| Negation | `!`, `not` | Logical NOT |
| Yesterday | `@`, `Y` | Value in previous state |
| Conjunction | `&`, `&&`, `and` | Logical AND |
| Disjunction | `\|`, `\|\|`, `or` | Logical OR |
| Implication | `->`, `implies` | If-then |
| Biconditional | `<->`, `iff` | If and only if |
| Since | `S`, `since` | Since temporal operator |

**Common patterns:**

```
# Once (eventually in past): TRUE S phi
TRUE S request

# Historically (always in past): !(TRUE S !phi)
!(TRUE S !valid)

# Response: request implies past event
response -> (TRUE S request)

# Safety with Since
done -> (confirmed S ready)
```

## Python API

```python
from pathlib import Path
from prove.core.monitor import EPLTLMonitor
from prove.parser.formula import parse_formula

# From files
monitor = EPLTLMonitor.from_files(
    property_file=Path("formula.prop"),
    trace_file=Path("trace.csv"),
    epsilon=2.0,
)
result = monitor.run_from_trace()

# Programmatic
formula = parse_formula("done -> (confirmed S ready)")
monitor = EPLTLMonitor(
    formula=formula,
    processes={"P1", "P2"},
    epsilon=2.0,
)
result = monitor.run(events)

print(result.satisfied)   # True/False
print(result.verdict)     # Human-readable verdict
print(result.statistics)  # Verification statistics
```

## Project Structure

```
prove/
├── cli.py                   # Command-line interface
├── core/                    # Core monitoring engine
│   ├── vector_clock.py      # Fidge-Mattern vector clocks
│   ├── event.py             # Event representation
│   ├── partial_order.py     # Partial order computation
│   ├── frontier.py          # Frontier (global state)
│   ├── cut.py               # History-closed event sets
│   ├── sliding_window.py    # Sliding window graph
│   ├── summary.py           # EPLTL summary evaluation
│   ├── monitor.py           # Main monitor orchestration
│   ├── clock_drift.py       # Epsilon-based ordering
│   └── message_queue.py     # Message queue tracking
├── parser/                  # Formula parsing
│   ├── lexer.py             # Lexical analyzer
│   ├── grammar.py           # EPLTL grammar (SLY-based)
│   ├── ast_nodes.py         # AST node definitions
│   └── formula.py           # Formula utilities
└── utils/                   # Utilities
    ├── trace_reader.py      # CSV trace file parser
    ├── logger.py             # Structured logging
    └── visualization.py     # Graph visualization
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=prove --cov-report=html

# Run specific component tests
pytest tests/unit/test_vector_clock.py -v
```

## License

MIT
