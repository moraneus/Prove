# PROVE — Partial oRder Verification Engine

Runtime verification of Existential Past Linear Temporal Logic (EPLTL) properties
over partial order executions of distributed systems.

## Overview

PROVE verifies temporal properties against distributed system execution traces
without requiring a total order. It works directly on the partial order defined by
vector clocks and bounded clock drift (epsilon), checking whether **at least one
valid linearization** of the execution satisfies a given past-time temporal property.

The tool implements a sliding window algorithm that combines Fidge-Mattern vector
clocks for causal ordering, bounded clock skew for additional temporal ordering,
and summary-based EPLTL evaluation across all possible execution paths.

Based on *"Runtime Verification of Linear Temporal Properties over Partial Order
Executions"* by Doron Peled et al.

## Features

- Offline verification of EPLTL properties over partial orders
- Vector clock and epsilon-based (bounded clock drift) event ordering
- Sliding window graph algorithm for efficient state space exploration
- Asynchronous message-passing with FIFO ordering
- ASCII visualization of partial order executions
- DOT/Graphviz export of the sliding window graph
- Detailed statistics and configurable debug output

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/moraneus/Prove.git
cd Prove
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify the installation:

```bash
python -m prove --version
```

## Quick Start

Create a property file `safety.prop`:

```
# If done, then working was true at some point in the past
done -> (TRUE S working)
```

Create a trace file `trace.csv`:

```csv
# system_processes: P1
eid,processes,vc,timestamp,props,event_type,msg_partner
e0,P1,P1:1,0.0,init,local,
e1,P1,P1:2,1.0,working,local,
e2,P1,P1:3,2.0,done,local,
```

Run the verification:

```bash
python -m prove -p safety.prop -t trace.csv
```

Output:

```
SATISFIED: Property holds for at least one linearization
```

## Usage

```
python -m prove -p PROPERTY -t TRACE [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-p, --property FILE` | EPLTL formula file (required) | — |
| `-t, --trace FILE` | Trace CSV file (required) | — |
| `-e, --epsilon N` | Maximum clock skew in time units | infinity |
| `-o, --output MODE` | Output level: `silent`, `normal`, `verbose` | `normal` |
| `-d, --debug N` | Debug level 0–3 | `0` |
| `--visualize [FILE]` | Generate graph visualization (DOT format) | — |
| `--visualize-ascii` | Print ASCII partial order diagram | — |
| `--full-graph` | Keep all graph nodes (disable pruning) | — |
| `--stats` | Print verification statistics | — |
| `--version` | Show version and exit | — |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Property SATISFIED |
| `1` | Property VIOLATED |
| `2` | Error (invalid input, file not found, parse error) |

## Trace File Format

CSV with one event per row. Optional directives in comments configure system-wide settings.

```csv
# system_processes: Client|Server
# epsilon: 2.0
eid,processes,vc,timestamp,props,event_type,msg_partner
iota_c,Client,Client:1;Server:0,0.0,idle,local,
iota_s,Server,Client:0;Server:1,0.0,idle,local,
c_send,Client,Client:2;Server:0,1.0,request,send,Server
s_recv,Server,Client:2;Server:2,2.0,busy,receive,Client
s_send,Server,Client:2;Server:3,3.0,response,send,Client
c_recv,Client,Client:3;Server:3,4.0,done,receive,Server
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `eid` | Yes | Unique event identifier |
| `processes` | Yes | Process ID this event belongs to |
| `vc` | Yes | Vector clock (`P1:2;P2:1;P3:0`) |
| `timestamp` | Yes | Global timestamp for epsilon-based ordering |
| `props` | No | Pipe-separated propositions (`ready\|done`), may be empty |
| `event_type` | No | `local`, `send`, or `receive` (default: `local`) |
| `msg_partner` | Conditional | Target process (send) or source process (receive) |

### Initial Events

Each process must have exactly one initial event with:
- `VC[p] = 1` for its own process
- `VC[q] = 0` for all other processes

Initial events are concurrent with each other and form the initial frontier.

## Property File Format

One EPLTL formula per file. Lines starting with `#` are comments.

### Operators

| Operator | Symbols | Description |
|----------|---------|-------------|
| True | `TRUE`, `true` | Always true |
| False | `FALSE`, `false` | Always false |
| Negation | `!`, `not`, `¬` | Logical NOT |
| Yesterday | `@`, `Y`, `prev` | True in previous state |
| Conjunction | `&`, `&&`, `and`, `∧` | Logical AND |
| Disjunction | `\|`, `\|\|`, `or`, `∨` | Logical OR |
| Implication | `->`, `implies`, `→` | If-then |
| Biconditional | `<->`, `iff`, `↔` | If and only if |
| Since | `S`, `since` | Since temporal operator |

### Precedence (highest to lowest)

1. `!`, `@` — unary (right-to-left)
2. `S` — since (right-to-left)
3. `&` — conjunction (left-to-right)
4. `\|` — disjunction (left-to-right)
5. `->` — implication (right-to-left)
6. `<->` — biconditional (left-to-right)

### Common Patterns

```bash
# Response: every response was preceded by a request
response -> (TRUE S request)

# Invariant: valid has always been true
!(TRUE S !valid)

# Precedence: alarm requires a prior warning
alarm -> (TRUE S warning)

# Safety with Since: confirmed held since ready became true
done -> (confirmed S ready)

# Yesterday: ready was true in the previous state
@ready
```

## Examples

```bash
# Basic property check
python -m prove -p examples/01_single_process_workflow/property.prop \
                -t examples/01_single_process_workflow/trace.csv

# Client-server with message passing
python -m prove -p examples/03_client_server_messages/property.prop \
                -t examples/03_client_server_messages/trace.csv

# Verbose output with statistics
python -m prove -p examples/05_since_operator/property.prop \
                -t examples/05_since_operator/trace.csv \
                -o verbose --stats

# ASCII visualization of partial order
python -m prove -p examples/03_client_server_messages/property.prop \
                -t examples/03_client_server_messages/trace.csv \
                --visualize-ascii

# Generate DOT graph file
python -m prove -p examples/01_single_process_workflow/property.prop \
                -t examples/01_single_process_workflow/trace.csv \
                --visualize graph.dot
```

## ASCII Visualization

The `--visualize-ascii` flag prints a timeline diagram of the partial order,
showing causal relationships between events across processes.

```
         Client                  Server
────────────────────────────────────────────────
(ε = inf)

     iota_c  {idle}          iota_s  {idle}
        (t=0.0)                 (t=0.0)
           │
           ↓
   c_send  {request}
        (t=1.0)
                                   │
                                   ↓
            ╰ c_send ≺ s_recv (VC) ─→
                             s_recv  {busy}
                                (t=2.0)
                                   │
                                   ↓
                        s_process  {processing}
                                (t=3.0)
                                   │
                                   ↓
                           s_send  {response}
                                (t=4.0)
           │
           ↓
            ← s_send ≺ c_recv (VC) ─╯
  c_recv  {satisfied}
        (t=5.0)

────────────────────────────────────────────────
Cross-process orderings:
  c_send ≺ s_recv  (VC)
  s_send ≺ c_recv  (VC)
```

Annotations on cross-process arrows indicate the ordering reason:

| Annotation | Meaning |
|------------|---------|
| `(VC)` | Vector clock ordering (message causality) |
| `(Δt=X.X>ε=Y.Y)` | Timestamp difference exceeds clock skew bound |

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

## Development

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=prove --cov-report=html

# Run specific tests
pytest tests/unit/test_vector_clock.py -v

# Format code
black prove/ tests/
isort prove/ tests/
```

Python 3.10+ required. Dependencies: `sly` (parser), `graphviz` (optional, visualization).

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
    ├── logger.py            # Structured logging
    └── visualization.py     # Graph visualization
```

## References

> Doron Peled et al., *"Runtime Verification of Linear Temporal Properties
> over Partial Order Executions."*

## License

See [LICENSE](LICENSE) file.
